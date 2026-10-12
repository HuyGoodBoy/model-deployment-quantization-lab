import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.util.*;

/** Desktop harness only; Android consumes MlsdPostProcessor without this class. */
public final class PostprocessBench {
    private static double number(String value) {
        if(value.equalsIgnoreCase("nan"))return Double.NaN;
        if(value.equalsIgnoreCase("inf"))return Double.POSITIVE_INFINITY;
        if(value.equalsIgnoreCase("-inf"))return Double.NEGATIVE_INFINITY;
        return Double.parseDouble(value);
    }
    private static List<MlsdPostProcessor.Segment> read(Path file) throws Exception {
        List<MlsdPostProcessor.Segment> lines=new ArrayList<>();
        List<String> rows=Files.readAllLines(file,StandardCharsets.UTF_8);
        for(int i=1;i<rows.size();i++) {
            if(rows.get(i).trim().isEmpty())continue;
            String[] c=rows.get(i).split(",");
            lines.add(new MlsdPostProcessor.Segment(number(c[0]),number(c[1]),number(c[2]),number(c[3])));
        }
        return lines;
    }
    private static double percentile(double[] values,double q) {
        double[] v=values.clone();Arrays.sort(v);double i=(v.length-1)*q;int lo=(int)i,hi=(int)Math.ceil(i);
        return v[lo]+(v[hi]-v[lo])*(i-lo);
    }
    private static String resultJson(MlsdPostProcessor.Result r) {
        StringBuilder s=new StringBuilder("{\"merged\":[");
        for(int i=0;i<r.merged.size();i++){if(i>0)s.append(',');s.append(Arrays.toString(r.merged.get(i).coordinates()));}
        s.append("],\"corners\":[");boolean comma=false;
        for(List<MlsdPostProcessor.Corner> group:r.corners)for(MlsdPostProcessor.Corner c:group) {
            if(comma)s.append(',');comma=true;
            s.append('[').append(c.type).append(',').append(c.x).append(',').append(c.y).append(',').append(c.i).append(',').append(c.j).append(']');
        }
        s.append("],\"boxes\":[");
        for(int i=0;i<r.boxes.size();i++) {
            if(i>0)s.append(',');MlsdPostProcessor.Box b=r.boxes.get(i);
            s.append("{\"score\":").append(b.score).append(",\"components\":")
             .append(Arrays.toString(new double[]{b.overlap,b.degree,b.area,b.center,b.length}));
            s.append(",\"edges\":").append(Arrays.toString(b.edges)).append(",\"corners\":[");
            for(int k=0;k<4;k++){if(k>0)s.append(',');s.append('[').append(b.corners[k].x).append(',').append(b.corners[k].y).append(']');}
            s.append("]}");
        }
        return s.append("],\"connection_checks\":").append(r.connectionChecks).append('}').toString();
    }
    private static String measuredJson(double[][] stages) {
        String[] names={"merge","corners","search","score","total"};StringBuilder s=new StringBuilder("{");
        for(int k=0;k<5;k++){if(k>0)s.append(',');s.append('"').append(names[k]).append("\":{\"median_ms\":")
            .append(percentile(stages[k],.5)).append(",\"p95_ms\":").append(percentile(stages[k],.95))
            .append(",\"samples_ms\":").append(Arrays.toString(stages[k])).append('}');}
        return s.append('}').toString();
    }
    private static String benchmark(List<MlsdPostProcessor.Segment> lines,int warmup,int runs,MlsdPostProcessor.Weights weights) {
        for(int i=0;i<warmup;i++){MlsdPostProcessor.detect(lines,false,weights);MlsdPostProcessor.detect(lines,true,weights);}
        double[][][] stages=new double[2][5][runs];
        for(int i=0;i<runs;i++) {
            for(int order=0;order<2;order++) {
                int mode=(i+order)%2;
                MlsdPostProcessor.Result r=MlsdPostProcessor.detect(lines,mode==1,weights);
                // Consume output and alternate measurement order to reduce mode-order bias.
                sink=r.boxes.size()+r.connectionChecks;
                for(int k=0;k<5;k++)stages[mode][k][i]=r.milliseconds[k];
            }
        }
        return "{\"naive\":"+measuredJson(stages[0])+",\"adjacency\":"+measuredJson(stages[1])+"}";
    }
    private static volatile long sink;
    private static void checkImageMapping() {
        List<MlsdPostProcessor.Segment> lines=Arrays.asList(
            new MlsdPostProcessor.Segment(20.3,30.4,220.8,30.4),
            new MlsdPostProcessor.Segment(220.8,30.4,220.8,220.7),
            new MlsdPostProcessor.Segment(220.8,220.7,20.3,220.7),
            new MlsdPostProcessor.Segment(20.3,220.7,20.3,30.4));
        MlsdPostProcessor.Box box=MlsdPostProcessor.detect(lines,true).boxes.get(0);
        double[][] image=MlsdPostProcessor.toImage(box,640,480);
        for(int k=0;k<4;k++)if(Math.abs(image[k][0]-box.corners[k].x*2.5)>1e-10
             ||Math.abs(image[k][1]-box.corners[k].y*1.875)>1e-10)
            throw new AssertionError("Non-square image mapping");
    }
    public static void main(String[] args) throws Exception {
        checkImageMapping();
        if(args.length<4)throw new IllegalArgumentException("input.csv output.json warmup runs [default|all-components]");
        List<MlsdPostProcessor.Segment> lines=read(Paths.get(args[0]));int warmup=Integer.parseInt(args[2]),runs=Integer.parseInt(args[3]);
        if(warmup<1||runs<1)throw new IllegalArgumentException("Positive counts required");
        MlsdPostProcessor.Weights w=args.length>4&&args[4].equals("all-components")?new MlsdPostProcessor.Weights(.7,1.95,1.86,.14,.6):MlsdPostProcessor.Weights.defaults();
        MlsdPostProcessor.Result naive=MlsdPostProcessor.detect(lines,false,w),adjacent=MlsdPostProcessor.detect(lines,true,w);
        if(!resultJson(naive).replace("\"connection_checks\":"+naive.connectionChecks,"\"connection_checks\":0")
                .equals(resultJson(adjacent).replace("\"connection_checks\":"+adjacent.connectionChecks,"\"connection_checks\":0")))
            throw new AssertionError("Adjacency changed output");
        String json="{\"scope\":\"desktop JVM post-processing only; no inference or Android\",\"warmup\":"+warmup+",\"runs\":"+runs+
            ",\"input_segments\":"+lines.size()+",\"naive\":"+resultJson(naive)+",\"adjacency\":"+resultJson(adjacent)+
            ",\"benchmark\":"+benchmark(lines,warmup,runs,w)+"}";
        Files.write(Paths.get(args[1]),json.getBytes(StandardCharsets.UTF_8));
        System.out.println(Paths.get(args[0]).getFileName()+": "+lines.size()+" -> "+adjacent.merged.size()+" merged, "+adjacent.boxes.size()+" boxes");
    }
}
