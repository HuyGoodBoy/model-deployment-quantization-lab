/*
 * M-LSD post-processing, adapted from NAVER utils.py, revision 453cafa09467.
 * Copyright 2021-present NAVER Corp. Licensed under Apache-2.0.
 * Java adaptation, bounds checks and adjacency search added on 2026-10-12.
 * License: daily/2026-10-09/experiments/03-mlsd/LICENSE-MLSD-NAVER.txt (repo root).
 */
import java.util.*;

/** Pure Java geometry core. Input and output coordinates are on the 256x256 map. */
public final class MlsdPostProcessor {
    private static final int ROWS = 513, COLS = 181, TOP_K = 200;
    private static final double EPS = 1e-10;

    public static final class Segment {
        public final double x1, y1, x2, y2;
        public Segment(double x1, double y1, double x2, double y2) {
            this.x1=x1; this.y1=y1; this.x2=x2; this.y2=y2;
        }
        public double length() { return Math.hypot(x1-x2, y1-y2); }
        public double[] coordinates() { return new double[]{x1,y1,x2,y2}; }
    }

    public static final class Corner {
        public final int x,y,i,j,type;
        private Corner(int x,int y,int i,int j,int type) {
            this.x=x; this.y=y; this.i=i; this.j=j; this.type=type;
        }
        private boolean has(int id) { return i==id || j==id; }
    }

    public static final class Box {
        public final Corner[] corners;
        public final int[] edges;
        public double score, overlap, degree, area, center, length;
        private Box(Corner[] corners,int[] edges) { this.corners=corners; this.edges=edges; }
    }

    public static final class Weights {
        public final double overlap,degree,area,center,length;
        public Weights(double overlap,double degree,double area,double center,double length) {
            this.overlap=overlap; this.degree=degree; this.area=area;
            this.center=center; this.length=length;
        }
        public static Weights defaults() { return new Weights(0,1.95,1.86,.14,0); }
    }

    public static final class Result {
        public final List<Segment> merged;
        public final List<List<Corner>> corners;
        public final List<Box> boxes;
        public final long connectionChecks;
        // merge, corners, search (including adjacency construction), scoring, total
        public final double[] milliseconds;
        private Result(List<Segment> merged,List<List<Corner>> corners,List<Box> boxes,
                       long checks,double[] ms) {
            this.merged=merged; this.corners=corners; this.boxes=boxes;
            this.connectionChecks=checks; this.milliseconds=ms;
        }
    }

    private static final class Bin {
        final int row,col,first;
        final List<Integer> members=new ArrayList<>();
        Bin(int row,int col,int first) { this.row=row;this.col=col;this.first=first; }
    }

    /** Source-compatible Hough bins, 5x5 peak suppression and endpoint-extrema merge. */
    public static List<Segment> merge(List<Segment> input) {
        Map<Integer,Bin> bins=new HashMap<>();
        double[] angles=new double[input.size()];
        for(int k=0;k<input.size();k++) {
            Segment s=input.get(k);
            if(!finite(s) || s.length()<1e-8) continue;
            double dx=s.x1-s.x2,dy=s.y1-s.y2,a=dy,b=-dx,c=a*s.x1+b*s.y1;
            double d=Math.abs(c)/Math.sqrt(a*a+b*b+EPS);
            double theta=Math.toDegrees(Math.atan2(dx,dy));
            if(theta<0) theta+=180;
            angles[k]=theta;
            int row=(int)Math.floor(d),col=(int)Math.floor(theta/2);
            if(row<0 || row>=ROWS || col<0 || col>=COLS) continue;
            int key=row*COLS+col;
            Bin bin=bins.get(key);
            if(bin==null) { bin=new Bin(row,col,k); bins.put(key,bin); }
            bin.members.add(k);
        }
        List<Bin> peaks=new ArrayList<>();
        for(Bin bin:bins.values()) {
            boolean peak=true;
            for(int dr=-2;dr<=2 && peak;dr++) for(int dc=-2;dc<=2;dc++) {
                Bin other=getBin(bins,bin.row+dr,bin.col+dc);
                if(other!=null && other.members.size()>bin.members.size()) { peak=false;break; }
            }
            if(peak) peaks.add(bin);
        }
        // TensorFlow top_k: descending counts, lower flattened index first for ties.
        peaks.sort((a,b)-> {
            int count=Integer.compare(b.members.size(),a.members.size());
            return count!=0?count:Integer.compare(a.row*COLS+a.col,b.row*COLS+b.col);
        });
        List<Segment> merged=new ArrayList<>();
        for(Bin peak:peaks.subList(0,Math.min(TOP_K,peaks.size()))) {
            double xmin=Double.POSITIVE_INFINITY,ymin=xmin;
            double xmax=Double.NEGATIVE_INFINITY,ymax=xmax;
            for(int dr=-2;dr<=2;dr++) for(int dc=-2;dc<=2;dc++) {
                Bin bin=getBin(bins,peak.row+dr,peak.col+dc);
                if(bin==null) continue;
                for(int index:bin.members) {
                    Segment s=input.get(index);
                    xmin=Math.min(xmin,Math.min(s.x1,s.x2));xmax=Math.max(xmax,Math.max(s.x1,s.x2));
                    ymin=Math.min(ymin,Math.min(s.y1,s.y2));ymax=Math.max(ymax,Math.max(s.y1,s.y2));
                }
            }
            Segment s=angles[peak.first]>=90?new Segment(xmin,ymax,xmax,ymin):new Segment(xmin,ymin,xmax,ymax);
            if(finite(s) && s.length()>1e-8) merged.add(s);
        }
        return merged;
    }

    private static Bin getBin(Map<Integer,Bin> bins,int row,int col) {
        // Deliberate hardening: no NumPy negative-index wrap or upper-bound overflow.
        if(row<0 || row>=ROWS || col<0 || col>=COLS) return null;
        return bins.get(row*COLS+col);
    }

    private static boolean finite(Segment s) {
        return Double.isFinite(s.x1)&&Double.isFinite(s.y1)&&Double.isFinite(s.x2)&&Double.isFinite(s.y2);
    }

    private static double[] intersection(Segment p,Segment q) {
        double a1=p.y1-p.y2,b1=p.x2-p.x1,c1=a1*p.x1+b1*p.y1;
        double a2=q.y1-q.y2,b2=q.x2-q.x1,c2=a2*q.x1+b2*q.y1;
        double det=a1*b2-a2*b1;
        if(Math.abs(det)<1e-12*p.length()*q.length()) return null;
        // Keep upstream epsilon and integer truncation for the compatibility experiment.
        double x=(c1*b2-c2*b1)/(det+EPS),y=(a1*c2-a2*c1)/(det+EPS);
        if(!Double.isFinite(x)||!Double.isFinite(y)||Math.abs(x)>Integer.MAX_VALUE||Math.abs(y)>Integer.MAX_VALUE) return null;
        return new double[]{(int)x,(int)y};
    }

    private static boolean nearEndpoint(Segment s,double x,double y) {
        double d1=Math.hypot(x-s.x1,y-s.y1),d2=Math.hypot(x-s.x2,y-s.y2);
        double min=Math.min(d1,d2),max=Math.max(d1,d2),len=s.length();
        return (max>=len && min<=len*.28)||(max<=len && min<=len*.45);
    }

    private static double direction(double x,double y) {
        double d=Math.toDegrees(Math.atan2(y,x)); return d<0?d+360:d;
    }

    private static int cornerType(double d1,double d2) {
        double lo=Math.min(d1,d2),hi=Math.max(d1,d2);
        if((lo>=0&&lo<=45&&hi>=45&&hi<=120)||(hi>=315&&lo>=45&&lo<=120)) return 0;
        if(lo>=45&&lo<=125&&hi>=125&&hi<=225) return 1;
        if(lo>=125&&lo<=225&&hi>=225&&hi<=315) return 2;
        if((lo>=0&&lo<=45&&hi>=225&&hi<=315)||(hi>=315&&lo>=225&&lo<=315)) return 3;
        return -1;
    }

    public static List<List<Corner>> corners(List<Segment> lines) {
        List<List<Corner>> groups=new ArrayList<>();for(int t=0;t<4;t++)groups.add(new ArrayList<>());
        for(int i=0;i<lines.size();i++)for(int j=i+1;j<lines.size();j++) {
            Segment a=lines.get(i),b=lines.get(j);double[] p=intersection(a,b);
            if(p==null||!nearEndpoint(a,p[0],p[1])||!nearEndpoint(b,p[0],p[1]))continue;
            double ax=(a.x1+a.x2)/2-p[0],ay=(a.y1+a.y2)/2-p[1];
            double bx=(b.x1+b.x2)/2-p[0],by=(b.y1+b.y2)/2-p[1];
            if(Math.hypot(ax,ay)<1e-12||Math.hypot(bx,by)<1e-12)continue;
            double d1=direction(ax,ay),d2=direction(bx,by),diff=Math.abs(d1-d2);
            if(diff>180)diff=360-diff;
            if(!(diff>60&&diff<120))continue;
            int type=cornerType(d1,d2);
            if(type>=0)groups.get(type).add(new Corner((int)p[0],(int)p[1],i,j,type));
        }
        return groups;
    }

    private static int common(Corner a,Corner b) {return b.has(a.i)?a.i:b.has(a.j)?a.j:-1;}

    /** Sorted neighbors preserve upstream candidate order and ties. */
    private static int[][] adjacency(List<Corner> from,List<Corner> to,int lineCount) {
        List<List<Integer>> byLine=new ArrayList<>();for(int i=0;i<lineCount;i++)byLine.add(new ArrayList<>());
        for(int k=0;k<to.size();k++){Corner c=to.get(k);byLine.get(c.i).add(k);byLine.get(c.j).add(k);}
        int[][] result=new int[from.size()][];
        for(int k=0;k<from.size();k++) {
            Corner c=from.get(k);List<Integer>a=byLine.get(c.i),b=byLine.get(c.j);
            int[] joined=new int[a.size()+b.size()];int ia=0,ib=0,n=0;
            while(ia<a.size()||ib<b.size()) {
                int next;
                if(ib==b.size()||(ia<a.size()&&a.get(ia)<b.get(ib)))next=a.get(ia++);
                else if(ia==a.size()||b.get(ib)<a.get(ia))next=b.get(ib++);
                else{next=a.get(ia++);ib++;}
                joined[n++]=next;
            }
            result[k]=Arrays.copyOf(joined,n);
        }
        return result;
    }

    private static int[][] allNeighbors(int from,int to) {
        int[] all=new int[to];for(int i=0;i<to;i++)all[i]=i;
        int[][] result=new int[from][];Arrays.fill(result,all);return result;
    }

    private static List<Box> search(List<List<Corner>> g,int n,boolean useAdjacency,long[] checks) {
        int[][] n01=useAdjacency?adjacency(g.get(0),g.get(1),n):allNeighbors(g.get(0).size(),g.get(1).size());
        int[][] n12=useAdjacency?adjacency(g.get(1),g.get(2),n):allNeighbors(g.get(1).size(),g.get(2).size());
        int[][] n23=useAdjacency?adjacency(g.get(2),g.get(3),n):allNeighbors(g.get(2).size(),g.get(3).size());
        List<Box> boxes=new ArrayList<>();
        for(int a=0;a<g.get(0).size();a++)for(int b:n01[a]) {
            Corner c0=g.get(0).get(a),c1=g.get(1).get(b);checks[0]++;int e01=common(c0,c1);if(e01<0)continue;
            for(int c:n12[b]) {
                Corner c2=g.get(2).get(c);checks[0]++;int e12=common(c1,c2);if(e12<0)continue;
                for(int d:n23[c]) {
                    Corner c3=g.get(3).get(d);checks[0]++;int e23=common(c2,c3);if(e23<0)continue;
                    checks[0]++;
                    for(int e30:new int[]{c3.i,c3.j})if(c0.has(e30))
                        boxes.add(new Box(new Corner[]{c0,c1,c2,c3},new int[]{e01,e12,e23,e30}));
                }
            }
        }
        return boxes;
    }

    private static double[] support(Corner c,int id,List<Segment> lines) {
        Segment s=lines.get(id);double d1=Math.hypot(c.x-s.x1,c.y-s.y1),d2=Math.hypot(c.x-s.x2,c.y-s.y2);
        return new double[]{Math.min(d1,d2),Math.max(d1,d2)>s.length()?1:-1};
    }

    private static double ratio(double a,double b) {return Math.min(a,b)/Math.max(a,b);}

    private static void score(List<Box> boxes,List<Segment> lines,Weights weights) {
        double maxOverlap=0;
        Iterator<Box> iterator=boxes.iterator();
        while(iterator.hasNext()) {
            Box box=iterator.next();double cover=0,perimeter=0,area2=0,cx=0,cy=0;
            double[] lengths=new double[4],degrees=new double[4];
            for(int k=0;k<4;k++) {
                Corner p=box.corners[k],q=box.corners[(k+1)%4],prev=box.corners[(k+3)%4];
                int id=box.edges[k];Segment s=lines.get(id);
                double[] start=support(p,id,lines),end=support(q,id,lines);
                cover+=s.length()+(start[1]>0?0:-start[0])+(end[1]>0?0:-end[0]);
                lengths[k]=s.length()+start[1]*start[0]+end[1]*end[0];perimeter+=lengths[k];
                double vx=prev.x-p.x,vy=prev.y-p.y,wx=q.x-p.x,wy=q.y-p.y;
                double dot=vx/(Math.hypot(vx,vy)+EPS)*(wx/(Math.hypot(wx,wy)+EPS))
                          +vy/(Math.hypot(vx,vy)+EPS)*(wy/(Math.hypot(wx,wy)+EPS));
                degrees[k]=Math.toDegrees(Math.acos(Math.max(-1,Math.min(1,dot))));
                area2+=(double)p.x*q.y-(double)p.y*q.x;cx+=p.x;cy+=p.y;
            }
            box.overlap=cover/perimeter;
            box.degree=(ratio(degrees[0],degrees[2])+ratio(degrees[1],degrees[3]))/2;
            box.length=(ratio(lengths[0],lengths[2])+ratio(lengths[1],lengths[3]))/2;
            box.area=Math.abs(area2)*.5/(256*256);
            box.center=Math.hypot(cx/4-128,cy/4-128)/(256/Math.sqrt(2));
            if(!Double.isFinite(box.overlap+box.degree+box.length+box.area+box.center)){iterator.remove();continue;}
            maxOverlap=Math.max(maxOverlap,box.overlap);
        }
        for(Box box:boxes) {
            box.overlap=maxOverlap>0?box.overlap/maxOverlap:0;
            box.score=weights.overlap*box.overlap+weights.degree*box.degree+weights.area*box.area
                      -weights.center*box.center+weights.length*box.length;
        }
        // Stable ordering for score ties; upstream NumPy argsort tie order is not guaranteed.
        boxes.sort((a,b)->Double.compare(b.score,a.score));
    }

    public static Result detect(List<Segment> input,boolean useAdjacency) {
        return detect(input,useAdjacency,Weights.defaults());
    }

    public static Result detect(List<Segment> input,boolean useAdjacency,Weights weights) {
        long t0=System.nanoTime();List<Segment> merged=merge(input);long t1=System.nanoTime();
        List<List<Corner>> corners=corners(merged);long t2=System.nanoTime();long[] checks={0};
        List<Box> boxes=search(corners,merged.size(),useAdjacency,checks);long t3=System.nanoTime();
        score(boxes,merged,weights);long t4=System.nanoTime();
        return new Result(merged,corners,boxes,checks[0],new double[]{(t1-t0)/1e6,(t2-t1)/1e6,(t3-t2)/1e6,(t4-t3)/1e6,(t4-t0)/1e6});
    }

    /** Direct-resize inverse. Letterbox/crop needs its own inverse transform. */
    public static double[][] toImage(Box box,int width,int height) {
        if(width<=0||height<=0)throw new IllegalArgumentException("Positive image dimensions required");
        double[][] points=new double[4][2];
        for(int k=0;k<4;k++){points[k][0]=box.corners[k].x*width/256.0;points[k][1]=box.corners[k].y*height/256.0;}
        return points;
    }
}
