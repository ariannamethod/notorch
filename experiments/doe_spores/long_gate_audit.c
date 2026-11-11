/* Read saved native traces and GGUF lineage metadata. No model is loaded or run.
 * Probabilities are normalized by each printed row's actual sum. Entropy is in
 * nats; effective seats is averaged per position, not exp(mean entropy).
 * Exact argmax ties choose the lowest slot, and tied rows are reported. */
#include "gguf.h"
#include <ctype.h>
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { E=6, ARMS=6, LAST=128, POINTS=16 };
static const char *arms[ARMS]={"key-awake","key-dream","key-replay","bell-awake","bell-dream","bell-replay"};
typedef struct {
    long n, winners[E], tied, p0_99;
    double max_gate, entropy, effective, mass[E], min_sum, max_sum, max_sum_error;
    double min_p0, max_p0, k, nonzero, consensus;
} Stats;
static Stats probes[ARMS][POINTS][2], waking[ARMS][LAST+1], private_votes[ARMS][LAST+1], baseline[2];
static int cycles[ARMS][LAST+1];
static long files_read=0, rows_read=0;
static double global_sum_min=INFINITY,global_sum_max=-INFINITY;
static char identity[65],source_host[65];

static void fail(const char *file,int line,const char *why) {
    fprintf(stderr,"long_gate_audit: %s:%d: %s\n",file,line,why); exit(1);
}
static int add(Stats *s,const double gates[E],int k,double consensus) {
    double sum=0;
    for(int i=0;i<E;i++) { if(!isfinite(gates[i]) || gates[i]<0) return -1; sum+=gates[i]; }
    if(!isfinite(sum) || fabs(sum-1)>1e-5 || sum<=0 || k<1 || k>E || !isfinite(consensus)) return -1;
    double p[E],h=0,max=-1; int winner=0,ties=0,positive=0;
    for(int i=0;i<E;i++) {
        p[i]=gates[i]/sum;
        if(p[i]>0) { h-=p[i]*log(p[i]); positive++; }
        if(p[i]>max) { max=p[i]; winner=i; ties=1; }
        else if(p[i]==max) ties++;
    }
    if(!s->n) { s->min_sum=s->max_sum=sum; s->min_p0=s->max_p0=p[0]; }
    s->min_sum=fmin(s->min_sum,sum); s->max_sum=fmax(s->max_sum,sum);
    s->max_sum_error=fmax(s->max_sum_error,fabs(sum-1));
    s->min_p0=fmin(s->min_p0,p[0]); s->max_p0=fmax(s->max_p0,p[0]);
    s->n++; s->max_gate+=max; s->entropy+=h; s->effective+=exp(h);
    s->winners[winner]++; s->tied+=ties>1; s->p0_99+=p[0]>=.99;
    s->k+=k; s->nonzero+=positive; s->consensus+=consensus;
    for(int i=0;i<E;i++) s->mass[i]+=p[i];
    return 0;
}
static void merge(Stats *a,const Stats *b) {
    if(!b->n) return;
    if(!a->n) { *a=*b; return; }
    a->n+=b->n; a->max_gate+=b->max_gate; a->entropy+=b->entropy; a->effective+=b->effective;
    a->tied+=b->tied; a->p0_99+=b->p0_99; a->k+=b->k; a->nonzero+=b->nonzero; a->consensus+=b->consensus;
    a->min_sum=fmin(a->min_sum,b->min_sum); a->max_sum=fmax(a->max_sum,b->max_sum);
    a->max_sum_error=fmax(a->max_sum_error,b->max_sum_error);
    a->min_p0=fmin(a->min_p0,b->min_p0); a->max_p0=fmax(a->max_p0,b->max_p0);
    for(int i=0;i<E;i++) { a->mass[i]+=b->mass[i]; a->winners[i]+=b->winners[i]; }
}
static void oracle(void) {
    Stats s={0}; double u[E]={1./6,1./6,1./6,1./6,1./6,1./6};
    if(add(&s,u,6,0) || fabs(s.max_gate-1./6)>1e-14 || fabs(s.entropy-log(6))>1e-14 ||
       fabs(s.effective-6)>1e-13 || s.winners[0]!=1 || s.tied!=1 || s.k!=6 || s.nonzero!=6) fail("oracle",1,"uniform");
    s=(Stats){0}; double hot[E]={0,0,0,1,0,0};
    if(add(&s,hot,1,1) || s.entropy!=0 || s.effective!=1 || s.max_gate!=1 ||
       s.winners[3]!=1 || s.tied || s.k!=1 || s.nonzero!=1) fail("oracle",2,"one-hot");
    s=(Stats){0}; double rounded[E]={.166666667,.166666667,.166666667,.166666667,.166666667,.166666667};
    if(add(&s,rounded,6,0) || fabs(s.effective-6)>1e-13 || fabs(s.mass[5]-1./6)>1e-14)
        fail("oracle",3,"normalization");
    double bad[E]={0},negative[E]={-1,2,0,0,0,0},nan[E]={NAN,0,0,0,0,0};
    if(!add(&s,bad,1,0) || !add(&s,negative,1,0) || !add(&s,nan,1,0)) fail("oracle",4,"invalid gates");
    fprintf(stderr,"oracle PASS: uniform H=ln6/effective6; onehot H=0/effective1; printed-sum normalization; invalid rows rejected\n");
}
static void header(FILE *out) {
    fputs("kind\tarm\tordinal\tprobe\tcycle\tpositions\tmean_max_gate\tmean_entropy_nats\tmean_effective_seats",out);
    for(int i=0;i<E;i++) fprintf(out,"\tmean_mass%d",i);
    for(int i=0;i<E;i++) fprintf(out,"\twinners%d",i);
    fputs("\ttied_rows\tp0_ge_0.99\tmin_p0\tmax_p0\tmin_raw_sum\tmax_raw_sum\tmax_sum_error\tmean_reported_k\tmean_nonzero\tmean_consensus\n",out);
}
static void row(FILE *out,const char *kind,const char *arm,int ordinal,int probe,int cycle,const Stats *s) {
    if(!s->n) fail(arm,ordinal,"empty metric group");
    fprintf(out,"%s\t%s\t%d\t%d\t%d\t%ld\t%.17g\t%.17g\t%.17g",kind,arm,ordinal,probe,cycle,s->n,
        s->max_gate/s->n,s->entropy/s->n,s->effective/s->n);
    for(int i=0;i<E;i++) fprintf(out,"\t%.17g",s->mass[i]/s->n);
    for(int i=0;i<E;i++) fprintf(out,"\t%ld",s->winners[i]);
    fprintf(out,"\t%ld\t%ld\t%.17g\t%.17g\t%.17g\t%.17g\t%.17g\t%.17g\t%.17g\t%.17g\n",s->tied,s->p0_99,
        s->min_p0,s->max_p0,s->min_sum,s->max_sum,s->max_sum_error,s->k/s->n,s->nonzero/s->n,s->consensus/s->n);
}
static FILE *input(const char *path) {
    FILE *f=fopen(path,"r"); if(!f) fail(path,0,strerror(errno)); files_read++; return f;
}
static FILE *output(const char *dir,const char *name) {
    char p[4096]; snprintf(p,sizeof p,"%s/%s",dir,name);
    FILE *f=fopen(p,"wx"); if(!f) fail(p,0,strerror(errno)); return f;
}
static void read_trace(const char *path,int arm,int ordinal,int score) {
    FILE *f=input(path); char *line=NULL; size_t cap=0; int nline=0,probe=-1,wake=0;
    int expected[2]={-1,-1}, seen[LAST+1]={0};
    while(getline(&line,&cap,f)>=0) {
        nline++; int id,tokens;
        if(sscanf(line,"probe=%d tokens=%d",&id,&tokens)==2) {
            if(!score || id<0 || id>1 || expected[id]>=0 || tokens<1) fail(path,nline,"probe header");
            probe=id; expected[id]=tokens; continue;
        }
        if(sscanf(line,"experience wake ordinal=%d",&id)==1) {
            if(score || id<ordinal || id>ordinal+7 || seen[id]++) fail(path,nline,"waking ordinal");
            wake=id; continue;
        }
        if(sscanf(line,"doe dream-start cycle=%d",&id)==1 || sscanf(line,"doe replay-start cycle=%d",&id)==1) {
            if(score || !wake || wake%8 || cycles[arm][wake]) fail(path,nline,"private cycle");
            cycles[arm][wake]=id; continue;
        }
        if(!strstr(line,"vote pos=")) continue;
        char kind[24]; int pos,k,end=0; double consensus,g[E];
        int got=sscanf(line,"doe %23s pos=%d k=%d consensus=%lf gates=%lf,%lf,%lf,%lf,%lf,%lf%n",
            kind,&pos,&k,&consensus,g,g+1,g+2,g+3,g+4,g+5,&end);
        if(got!=10) fail(path,nline,"gate row syntax");
        for(char *p=line+end;*p;p++) if(!isspace((unsigned char)*p)) fail(path,nline,"gate trailing text");
        Stats *s=NULL;
        if(score) {
            if(strcmp(kind,"vote") || probe<0) fail(path,nline,"probe vote class");
            s=arm<0 ? &baseline[probe] : &probes[arm][ordinal/8-1][probe];
        } else if(!strcmp(kind,"vote")) {
            if(!wake) fail(path,nline,"waking context");
            s=&waking[arm][wake];
        } else {
            const char *want=arm%3==1 ? "dream-vote" : arm%3==2 ? "replay-vote" : "";
            if(strcmp(kind,want) || !wake || !cycles[arm][wake]) fail(path,nline,"private vote class");
            s=&private_votes[arm][wake];
        }
        if(pos!=s->n || add(s,g,k,consensus)) fail(path,nline,"non-sequential, nonfinite, negative or nonunit gate row");
        double sum=0; for(int i=0;i<E;i++) sum+=g[i];
        global_sum_min=fmin(global_sum_min,sum); global_sum_max=fmax(global_sum_max,sum); rows_read++;
    }
    if(ferror(f)) fail(path,nline,"read error");
    free(line); fclose(f);
    if(score) {
        for(int p=0;p<2;p++) {
            Stats *s=arm<0 ? &baseline[p] : &probes[arm][ordinal/8-1][p];
            if(s->n!=expected[p]) fail(path,nline,"missing probe positions");
        }
    } else {
        for(int n=ordinal;n<ordinal+8;n++) if(seen[n]!=1 || !waking[arm][n].n) fail(path,nline,"incomplete block");
        if((arm%3!=0)!=!!private_votes[arm][ordinal+7].n) fail(path,nline,"private cycle coverage");
    }
}
static void lineage(const char *path,FILE *out,const char *arm,int ordinal) {
    FILE *f=input(path); char *line=NULL; size_t cap=0; int lineno=0,found=0;
    while(getline(&line,&cap,f)>=0) {
        lineno++; if(lineno==1) continue;
        char *p=line,*field[7];
        for(int i=0;i<7;i++) { field[i]=strsep(&p,"\t\r\n"); if(!field[i]) fail(path,lineno,"lineage fields"); }
        if(strcmp(field[6],"0")) continue;
        if(found++ || strlen(field[0])!=64 || strlen(field[2])!=64 || strcmp(field[4],"0")) fail(path,lineno,"living seat0");
        if(!*identity) { strcpy(identity,field[0]); strcpy(source_host,field[2]); }
        if(strcmp(identity,field[0]) || strcmp(source_host,field[2])) fail(path,lineno,"seat0 identity changed");
        fprintf(out,"%s\t%d\t%s\t%s\t%s\t%s\t%s\n",arm,ordinal,field[0],field[2],field[3],field[4],field[5]);
    }
    free(line); fclose(f); if(found!=1) fail(path,0,"seat0 absent");
}
static const char *strkey(const gguf_file *g,const char *key) {
    const gguf_kv *v=gguf_get_kv(g,key); return v && v->type==8 ? v->val.str : "-";
}
static void parent(const char *path,FILE *out) {
    gguf_file *g=gguf_open(path); if(!g) fail(path,0,"parent GGUF");
    int n=0,ns=0; char **entries=gguf_read_fixed_str_array(path,"doe.lineage.entries",174,&n);
    int32_t *seats=gguf_read_i32_array(path,"doe.lineage.seats",&ns);
    if(!entries || !seats || ns!=E || seats[0]<1 || seats[0]>n) fail(path,0,"parent seat ledger");
    const char *e=entries[seats[0]-1];
    char id[65]={0},origin[65]={0}; memcpy(id,e+45,64); memcpy(origin,e+110,64);
    if(strcmp(identity,id) || strcmp(source_host,origin)) fail(path,0,"parent identity differs");
    fprintf(out,"%s\t%s\t%s\t%s\t%s\t%s\n",path,strkey(g,"doe.source_arch"),strkey(g,"doe.host_sha256"),id,origin,e);
    for(int i=0;i<n;i++) free(entries[i]);
    free(entries); free(seats); gguf_close(g);
}
int main(int argc,char **argv) {
    oracle();
    if(argc==2 && !strcmp(argv[1],"--selftest")) return 0;
    if(argc!=6) { fputs("usage: long_gate_audit MAIN OUTDIR PARENT_SCORES_TRACE OMNI_SPORE GEMMA_SPORE\n",stderr); return 2; }
    char path[4096];
    read_trace(argv[3],-1,0,1);
    for(int a=0;a<ARMS;a++) for(int p=0;p<POINTS;p++) {
        snprintf(path,sizeof path,"%s/%s-%d-scores.trace",argv[1],arms[a],8*(p+1)); read_trace(path,a,8*(p+1),1);
        snprintf(path,sizeof path,"%s/%s-block-%d.trace",argv[1],arms[a],8*p+1); read_trace(path,a,8*p+1,0);
    }
    FILE *pr=output(argv[2],"gate-probes.tsv"),*wk=output(argv[2],"gate-waking.tsv"),*pv=output(argv[2],"gate-private.tsv"),
         *se=output(argv[2],"gate-probe-series.tsv"),*to=output(argv[2],"gate-totals.tsv"),*li=output(argv[2],"gate-seat0-lineage.tsv");
    header(pr); header(wk); header(pv); header(se); header(to);
    fputs("arm\tordinal\tidentity\tsource_host\tidentity_turn\tdeath_turn\tkind\n",li);
    Stats all={0};
    for(int p=0;p<2;p++) { row(pr,"probe","parent",0,p,0,&baseline[p]); merge(&all,&baseline[p]); }
    row(se,"probe","parent",0,-1,0,&all); row(to,"probe","parent",0,-1,0,&all);
    for(int a=0;a<ARMS;a++) {
        Stats totalp={0},totalw={0},totals={0};
        for(int p=0;p<POINTS;p++) {
            Stats combined={0};
            for(int j=0;j<2;j++) { row(pr,"probe",arms[a],8*(p+1),j,0,&probes[a][p][j]); merge(&combined,&probes[a][p][j]); }
            row(se,"probe",arms[a],8*(p+1),-1,0,&combined); merge(&totalp,&combined);
            snprintf(path,sizeof path,"%s/%s-%d-lineage.txt",argv[1],arms[a],8*(p+1)); lineage(path,li,arms[a],8*(p+1));
        }
        for(int n=1;n<=LAST;n++) {
            row(wk,"wake",arms[a],n,-1,0,&waking[a][n]); merge(&totalw,&waking[a][n]);
            if(private_votes[a][n].n) {
                row(pv,a%3==1 ? "dream" : "replay",arms[a],n,-1,cycles[a][n],&private_votes[a][n]);
                merge(&totals,&private_votes[a][n]);
            }
        }
        row(to,"probe",arms[a],-1,-1,0,&totalp); row(to,"wake",arms[a],-1,-1,0,&totalw);
        if(totals.n) row(to,a%3==1 ? "dream" : "replay",arms[a],-1,-1,0,&totals);
    }
    FILE *pa=output(argv[2],"gate-parent-lineage.tsv");
    fputs("file\tcurrent_arch\tcurrent_host\tseat0_identity\tseat0_source_host\tlineage_record\n",pa);
    parent(argv[4],pa); parent(argv[5],pa);
    FILE *outs[]={pr,wk,pv,se,to,li,pa};
    for(unsigned i=0;i<sizeof(outs)/sizeof(*outs);i++) if(fclose(outs[i])) fail("output",0,"close");
    fprintf(stderr,"PASS trace rows=%ld raw_sum_min=%.17g raw_sum_max=%.17g; 194 probes,768 waking groups,64 private groups;96 lineage checkpoints and2parents share seat0=%s\n",
        rows_read,global_sum_min,global_sum_max,identity);
    return 0;
}
