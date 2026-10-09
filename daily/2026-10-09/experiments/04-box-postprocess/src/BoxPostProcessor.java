import java.util.*;
import java.nio.file.*;
import java.nio.charset.StandardCharsets;

/** Pure Java prototype for Android: quadrilaterals from already merged line segments.
 * This changes upstream candidate generation; it is not a verified drop-in NAVER port.
 * No TensorFlow, OpenCV or desktop-only math dependency is required by detect().
 */
public final class BoxPostProcessor {
    public static final class Segment {
        final double x1,y1,x2,y2,score;
        public Segment(double a,double b,double c,double d,double s) {x1=a;y1=b;x2=c;y2=d;score=s;}
        double length() { return Math.hypot(x2-x1,y2-y1); }
    }
    public static final class Box {
        public final double[][] corners;
        public final double score;
        Box(double[][] p,double s) {corners=p;score=s;}
    }
    static double cross(double ax,double ay,double bx,double by) {return ax*by-ay*bx;}
    public static double[] intersection(Segment a,Segment b) {
        double ax=a.x2-a.x1,ay=a.y2-a.y1,bx=b.x2-b.x1,by=b.y2-b.y1;
        double al=a.length(),bl=b.length(),det=cross(ax,ay,bx,by);
        if(al<1e-8 || bl<1e-8 || Math.abs(det)<1e-8*al*bl) return null;
        double cosine=Math.abs((ax*bx+ay*by)/(al*bl));
        if(cosine>0.5) return null; // acute angle must be at least 60 degrees
        double t=cross(b.x1-a.x1,b.y1-a.y1,bx,by)/det;
        double u=cross(b.x1-a.x1,b.y1-a.y1,ax,ay)/det;
        // bounded extrapolation accommodates small endpoint gaps
        if(t<-.1 || t>1.1 || u<-.1 || u>1.1) return null;
        return new double[]{a.x1+t*ax,a.y1+t*ay};
    }
    public static List<Box> detect(List<Segment> source,double width,double height) {
        List<Segment> lines=new ArrayList<>();
        for(Segment s:source) {
            if(Double.isFinite(s.x1+s.y1+s.x2+s.y2+s.score) && s.length()>1e-6) lines.add(s);
        }
        lines.sort((a,b)->Double.compare(b.score,a.score));
        if(lines.size()>64) lines=new ArrayList<>(lines.subList(0,64));
        int n=lines.size(); double[][][] points=new double[n][n][];
        for(int i=0;i<n;i++) for(int j=i+1;j<n;j++) points[i][j]=points[j][i]=intersection(lines.get(i),lines.get(j));
        List<Box> boxes=new ArrayList<>(); Set<String> seen=new HashSet<>();
        for(int a=0;a<n;a++) for(int b=a+1;b<n;b++) {
            if(points[a][b]==null) continue;
            for(int c=0;c<n;c++) {
                if(c==a || c==b || points[b][c]==null) continue;
                for(int d=0;d<n;d++) {
                    if(d==a || d==b || d==c || points[c][d]==null || points[d][a]==null) continue;
                    int[] ids={a,b,c,d}; Arrays.sort(ids);
                    String key=Arrays.toString(ids);
                    if(seen.contains(key)) continue;
                    double[][] p={points[a][b],points[b][c],points[c][d],points[d][a]};
                    double area=0,orientation=0; boolean valid=true;
                    for(int k=0;k<4;k++) {
                        double[] v=p[k],w=p[(k+1)%4],z=p[(k+2)%4];
                        double turn=cross(w[0]-v[0],w[1]-v[1],z[0]-w[0],z[1]-w[1]);
                        if(Math.abs(turn)<1e-6 || (orientation!=0 && turn*orientation<0)) valid=false;
                        orientation=turn;
                        area+=v[0]*w[1]-v[1]*w[0];
                        if(v[0]<-.1*width || v[0]>1.1*width || v[1]<-.1*height || v[1]>1.1*height) valid=false;
                    }
                    area=Math.abs(area)/2;
                    if(!valid || area<25 || area>1.21*width*height) continue;
                    seen.add(key);
                    double score=area/(width*height)+(lines.get(a).score+lines.get(b).score+lines.get(c).score+lines.get(d).score)/4;
                    boxes.add(new Box(p,score));
                }
            }
        }
        boxes.sort((a,b)->Double.compare(b.score,a.score));
        return boxes.size()>20 ? new ArrayList<>(boxes.subList(0,20)) : boxes;
    }
    static void require(boolean condition,String description) {
        if(!condition) throw new AssertionError(description);
    }
    static void tests() {
        List<Segment> rectangle=Arrays.asList(new Segment(10,20,110,20,1),new Segment(110,20,110,80,1),
                new Segment(110,80,10,80,1),new Segment(10,80,10,20,1));
        List<Box> result=detect(rectangle,200,100);
        require(result.size()==1,"Known rectangle must yield one quadrilateral");
        require(Math.abs(result.get(0).score-1.3)<1e-9,"Area scoring must use non-square image dimensions");
        require(intersection(rectangle.get(0),rectangle.get(2))==null,"Parallel lines");
        require(intersection(new Segment(0,0,0,0,1),rectangle.get(0))==null,"Zero length segment");
        require(detect(Collections.emptyList(),200,100).isEmpty(),"Empty input");
        require(intersection(new Segment(0,0,1,0,1),new Segment(5,0,5,1,1))==null,"Far intersection");
        System.out.println("5 geometry checks passed");
    }
    public static void main(String[] args) throws Exception {
        tests();
        if(args.length<2) return;
        List<Segment> lines=new ArrayList<>();
        List<String> rows=Files.readAllLines(Paths.get(args[0]),StandardCharsets.UTF_8);
        for(int i=1;i<rows.size();i++) {
            String[] c=rows.get(i).split(","); if(c.length!=5) continue;
            lines.add(new Segment(Double.parseDouble(c[0]),Double.parseDouble(c[1]),Double.parseDouble(c[2]),Double.parseDouble(c[3]),Double.parseDouble(c[4])));
        }
        for(int i=0;i<30;i++) detect(lines,512,512);
        double[] times=new double[200]; List<Box> result=null;
        for(int i=0;i<times.length;i++) {long start=System.nanoTime(); result=detect(lines,512,512);times[i]=(System.nanoTime()-start)/1e6;}
        double[] ordered=times.clone(); Arrays.sort(ordered);
        StringBuilder json=new StringBuilder("{\"scope\":\"desktop JVM geometry prototype, not Android measurement\",\"input_segments\":").append(lines.size());
        json.append(",\"candidate_limit\":64,\"warmup\":30,\"runs\":200,\"median_ms\":").append((ordered[99]+ordered[100])/2);
        json.append(",\"samples_ms\":").append(Arrays.toString(times)).append(",\"boxes\":[");
        for(int i=0;i<result.size();i++) {if(i>0)json.append(',');Box b=result.get(i);json.append("{\"score\":").append(b.score).append(",\"corners\":").append(Arrays.deepToString(b.corners)).append('}');}
        json.append("]}"); Files.write(Paths.get(args[1]),json.toString().getBytes(StandardCharsets.UTF_8));
        System.out.println("Saved "+result.size()+" boxes");
    }
}
