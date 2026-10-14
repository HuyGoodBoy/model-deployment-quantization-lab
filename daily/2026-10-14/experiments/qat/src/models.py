"""Weight-preserving DistilBERT definition and explicit W8A8 fake-quant wrappers."""

import copy
import itertools
import math

import numpy as np
import torch
from torch import nn
from torch.ao.quantization import FakeQuantize, MinMaxObserver, PerChannelMinMaxObserver
from torch.nn import functional as F


class Embeddings(nn.Module):
    def __init__(self):
        super().__init__()
        self.word_embeddings = nn.Embedding(30522, 768, padding_idx=0)
        self.position_embeddings = nn.Embedding(512, 768)
        self.LayerNorm = nn.LayerNorm(768, eps=1e-12)
        self.dropout = nn.Dropout(0.1)

    def forward(self, ids):
        positions = torch.arange(ids.shape[1], device=ids.device)[None]
        return self.dropout(
            self.LayerNorm(self.word_embeddings(ids) + self.position_embeddings(positions))
        )


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_lin = nn.Linear(768, 768)
        self.k_lin = nn.Linear(768, 768)
        self.v_lin = nn.Linear(768, 768)
        self.out_lin = nn.Linear(768, 768)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, mask):
        batch, length, _ = x.shape

        def heads(v):
            return v.view(batch, length, 12, 64).transpose(1, 2)

        q = heads(self.q_lin(x)) / math.sqrt(64)
        k = heads(self.k_lin(x))
        v = heads(self.v_lin(x))
        scores = torch.matmul(q, k.transpose(-2, -1))
        scores = scores.masked_fill(mask[:, None, None, :] == 0, -1e4)
        weights = self.dropout(torch.softmax(scores, dim=-1))
        context = torch.matmul(weights, v).transpose(1, 2).contiguous().view(batch, length, 768)
        return self.out_lin(context)


class FFN(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(768, 3072)
        self.lin2 = nn.Linear(3072, 768)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        return self.dropout(self.lin2(F.gelu(self.lin1(x), approximate="none")))


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = Attention()
        self.sa_layer_norm = nn.LayerNorm(768, eps=1e-12)
        self.ffn = FFN()
        self.output_layer_norm = nn.LayerNorm(768, eps=1e-12)

    def forward(self, x, mask):
        x = self.sa_layer_norm(x + self.attention(x, mask))
        return self.output_layer_norm(x + self.ffn(x))


class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.ModuleList([Block() for _ in range(6)])

    def forward(self, x, mask):
        for layer in self.layer:
            x = layer(x, mask)
        return x


class DistilBert(nn.Module):
    def __init__(self):
        super().__init__()
        self.embeddings = Embeddings()
        self.transformer = Transformer()

    def forward(self, ids, mask):
        return self.transformer(self.embeddings(ids), mask)


class TextClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.distilbert = DistilBert()
        self.pre_classifier = nn.Linear(768, 768)
        self.classifier = nn.Linear(768, 2)
        self.dropout = nn.Dropout(0.2)

    def forward(self, ids, mask):
        x = self.distilbert(ids, mask)[:, 0]
        return self.classifier(self.dropout(F.relu(self.pre_classifier(x))))


def load_text(path):
    model = TextClassifier()
    with np.load(path, allow_pickle=False) as weights:
        state = {name: torch.from_numpy(weights[name].copy()) for name in weights.files}
    model.load_state_dict(state, strict=True)
    return model.eval()


def activation_quant():
    return FakeQuantize(
        observer=MinMaxObserver,
        quant_min=-128,
        quant_max=127,
        dtype=torch.qint8,
        qscheme=torch.per_tensor_affine,
    )


class QatOperation(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer
        self.activation_fake_quant = activation_quant()
        self.weight_fake_quant = FakeQuantize(
            observer=PerChannelMinMaxObserver,
            quant_min=-127,
            quant_max=127,
            dtype=torch.qint8,
            qscheme=torch.per_channel_symmetric,
            ch_axis=0,
        )

    def forward(self, x):
        x = self.activation_fake_quant(x)
        w = self.weight_fake_quant(self.layer.weight)
        if isinstance(self.layer, nn.Linear):
            return F.linear(x, w, self.layer.bias)
        return F.conv2d(
            x,
            w,
            self.layer.bias,
            self.layer.stride,
            self.layer.padding,
            self.layer.dilation,
            self.layer.groups,
        )


class ActiveMlsd(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)[:, :5]


class QuantizedOutput(nn.Module):
    def __init__(self, model, task):
        super().__init__()
        self.model = model
        self.task = task
        self.center_quant = activation_quant()
        self.regression_quant = activation_quant()

    def forward(self, *args):
        x = self.model(*args)
        if self.task == "text":
            return self.center_quant(x)
        return torch.cat([self.center_quant(x[:, :1]), self.regression_quant(x[:, 1:5])], dim=1)


def fuse_batchnorm(model):
    count = 0
    for parent in list(model.modules()):
        if not isinstance(parent, nn.Sequential):
            continue
        children = list(parent.named_children())
        for (name, a), (next_name, b) in itertools.pairwise(children):
            if isinstance(a, nn.Conv2d) and isinstance(b, nn.BatchNorm2d):
                parent._modules[name] = torch.nn.utils.fuse_conv_bn_eval(a, b)
                parent._modules[next_name] = nn.Identity()
                count += 1
    return count


def prepare_qat(teacher, task):
    model = copy.deepcopy(teacher).eval()
    fused = fuse_batchnorm(model) if task == "mlsd" else 0
    names = []

    def wrap(parent, prefix=""):
        for name, child in list(parent.named_children()):
            full = prefix + name
            if isinstance(child, (nn.Conv2d, nn.Linear)):
                parent._modules[name] = QatOperation(child)
                names.append(full)
            else:
                wrap(child, full + ".")

    wrap(model)
    if task == "text":
        for parameter in model.distilbert.embeddings.parameters():
            parameter.requires_grad = False
    return QuantizedOutput(model, task), {
        "fused_batchnorms": fused,
        "fake_quantized_layers": names,
        "weights": "signed INT8, per-output-channel symmetric (-127..127)",
        "activations": "signed INT8, per-tensor affine (-128..127), MinMax observer",
        "float_regions": "bias, residual/add, resize, normalization; text embedding/attention MatMul/Softmax/mask",
        "scope": "selective W8A8 fake-quant training pilot; not an exported full INT8 model",
    }
