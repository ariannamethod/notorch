/* Read-only measurements of an explicit body-memory recall. All three spores
 * already belong to HOST: no decoder, transfer, calibration or learning runs.
 * The scalar oracle below does not call the projector being audited. */
#include "harness/experts_internal.h"
#include "harness/spore_frame.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct { const char *name; uint32_t type; } metadata_field;
static const metadata_field fields[] = {
    {"general.architecture",8}, {"doe.host_sha256",8}, {"doe.version",4},
    {"doe.layer",4}, {"doe.rng",4}, {"doe.turns",10}, {"doe.elections",10},
    {"doe.consensus",6}, {"doe.dream.cycles",10}, {"doe.dream.choices",10},
    {"doe.dream.elections",10}, {"doe.dream.seen",4},
    {"doe.replay.cycles",10}, {"doe.replay.choices",10},
    {"doe.replay.elections",10}, {"doe.replay.prefill",10},
    {"doe.replay.stops",10}, {"doe.replay.last",8},
    {"doe.lineage.version",4}, {"doe.lineage.entries",9}, {"doe.lineage.seats",9},
    {"doe.frame",8}, {"doe.source_arch",8}, {"doe.norm_sha256",8},
    {"doe.width",4}, {"doe.depth",4}
};

static int known_metadata(const gguf_file *g) {
    if (g->n_kv != (uint64_t)g->n_kv_parsed) return 0;
    for (int i=0; i<g->n_kv_parsed; i++) {
        const gguf_kv *kv=g->kv+i;
        size_t j=0;
        while (j<sizeof(fields)/sizeof(*fields) && strcmp(kv->key,fields[j].name)) j++;
        if (j==sizeof(fields)/sizeof(*fields) || kv->type!=fields[j].type) {
            fprintf(stderr,"recall audit: unexpected metadata %s in %s\n",kv->key,g->path);
            return 0;
        }
        for (int k=0; k<i; k++) if (!strcmp(kv->key,g->kv[k].key)) return 0;
    }
    return 1;
}

static int scalar_equal(const gguf_kv *a, const gguf_kv *b) {
    if (!a || !b) return a==b;
    if (a->type!=b->type) return 0;
    switch (a->type) {
    case 4: return a->val.u32==b->val.u32;
    case 6: return !memcmp(&a->val.f32,&b->val.f32,sizeof(float));
    case 8: return !strcmp(a->val.str,b->val.str);
    case 10: return a->val.u64==b->val.u64;
    default: return 0;
    }
}

static int metadata_preserved(const gguf_file *before, const gguf_file *after) {
    for (size_t i=0; i<sizeof(fields)/sizeof(*fields); i++) {
        const char *name=fields[i].name;
        /* Exact -> portable adds a measured frame; legacy -> current adds the
         * adopted ledger. The loaded ledger is compared field by field below. */
        if (!strcmp(name,"doe.version") || !strncmp(name,"doe.lineage.",12) ||
            !strcmp(name,"doe.frame") || !strcmp(name,"doe.source_arch") ||
            !strcmp(name,"doe.norm_sha256") || !strcmp(name,"doe.width") ||
            !strcmp(name,"doe.depth")) continue;
        if (!scalar_equal(gguf_get_kv(before,name),gguf_get_kv(after,name))) {
            fprintf(stderr,"recall audit: changed metadata %s\n",name);
            return 0;
        }
    }
    return 1;
}

static int state_preserved(const nt_experts *a, const nt_experts *b) {
#define SAME(member) do { if (memcmp(&a->member,&b->member,sizeof(a->member))) { \
    fprintf(stderr,"recall audit: changed state %s\n",#member); return 0; } } while (0)
    SAME(dim); SAME(layer); SAME(depth); SAME(host); SAME(arch);
    SAME(rng); SAME(turns); SAME(elections); SAME(consensus);
    SAME(sleep.cycles); SAME(sleep.choices); SAME(sleep.elections); SAME(sleep.seen);
    SAME(replay.cycles); SAME(replay.choices); SAME(replay.elections);
    SAME(replay.prefill); SAME(replay.stops); SAME(replay.last);
    SAME(alive); SAME(vitality); SAME(frequency); SAME(cursor);
    SAME(age); SAME(low); SAME(history); SAME(seat_ancestor); SAME(ancestor_count);
    SAME(count); SAME(observed); SAME(choices); SAME(hits); SAME(signal);
#undef SAME
    for (int i=0; i<a->ancestor_count; i++) {
        const nt_lineage_entry *x=a->ancestors+i,*y=b->ancestors+i;
        if (strcmp(x->id,y->id) || strcmp(x->source_host,y->source_host) ||
            x->turn!=y->turn || x->death_turn!=y->death_turn ||
            x->parent!=y->parent || x->kind!=y->kind) {
            fprintf(stderr,"recall audit: changed ledger entry %d\n",i);
            return 0;
        }
    }
    return 1;
}

typedef struct {
    char id[65];
    const char *kind;
    int slot, local_slot, floats, coordinates, unchanged;
    double incoming_norm, local_norm, returned_norm, incoming_delta, local_delta;
    double coordinate_error, orthogonal_error, formula_error, formula_tolerance;
} measurement;

static double maxabs(double a,double b) { return fmax(a,fabs(b)); }

static int measure(const nt_experts *incoming, const nt_experts *local,
                   const nt_experts *returned, const float *q,
                   int slot,int from,int kind,measurement *out) {
    static const char *names[]={"A","B","vote"};
    const float *vbase=kind==0 ? incoming->a : kind==1 ? incoming->b : incoming->vote;
    const float *ubase=kind==0 ? local->a : kind==1 ? local->b : local->vote;
    const float *rbase=kind==0 ? returned->a : kind==1 ? returned->b : returned->vote;
    int width=incoming->dim, rank=kind==2 ? 1 : RANK;
    size_t count=(size_t)width*rank;
    const float *v=vbase+(size_t)slot*count,*r=rbase+(size_t)slot*count;
    const float *u=from>=0 ? ubase+(size_t)from*count : NULL;
    nt_expert_identity id;
    if (nt_experts_seat_identity(incoming,slot,&id)) return 0;
    *out=(measurement){.kind=names[kind],.slot=slot,.local_slot=from,
        .floats=width*rank,.coordinates=NT_SPORE_ANCHORS*rank,
        .unchanged=!memcmp(v,r,count*sizeof(float))};
    snprintf(out->id,sizeof(out->id),"%s",id.id[0] ? id.id : "-");
    if (from<0 && !out->unchanged) {
        fprintf(stderr,"recall audit: unmatched slot %d %s changed\n",slot,names[kind]);
        return 0;
    }
    double expected_max=0;
    for (int vector=0; vector<rank; vector++) {
        double cv[NT_SPORE_ANCHORS]={0},cu[NT_SPORE_ANCHORS]={0};
        double cr[NT_SPORE_ANCHORS]={0};
        for (int k=0; k<NT_SPORE_ANCHORS; k++) {
            for (int i=0; i<width; i++) {
                size_t at=kind==0 ? (size_t)i*rank+vector : (size_t)vector*width+i;
                double axis=q[(size_t)k*width+i];
                cv[k]+=axis*v[at]; cr[k]+=axis*r[at];
                if (u) cu[k]+=axis*u[at];
            }
            out->coordinate_error=maxabs(out->coordinate_error,cr[k]-cv[k]);
        }
        for (int i=0; i<width; i++) {
            size_t at=kind==0 ? (size_t)i*rank+vector : (size_t)vector*width+i;
            double expected=u ? u[at] : v[at];
            double orthogonal_delta=u ? (double)r[at]-u[at] : 0;
            for (int k=0; u && k<NT_SPORE_ANCHORS; k++) {
                double axis=q[(size_t)k*width+i];
                expected+=axis*(cv[k]-cu[k]);
                orthogonal_delta-=axis*(cr[k]-cu[k]);
            }
            if (!isfinite(expected) || !isfinite(orthogonal_delta) ||
                !isfinite(v[at]) || !isfinite(r[at]) || (u && !isfinite(u[at]))) return 0;
            expected_max=maxabs(expected_max,expected);
            out->formula_error=maxabs(out->formula_error,r[at]-expected);
            out->orthogonal_error=maxabs(out->orthogonal_error,orthogonal_delta);
            out->incoming_norm+=(double)v[at]*v[at];
            out->returned_norm+=(double)r[at]*r[at];
            out->incoming_delta+=((double)r[at]-v[at])*((double)r[at]-v[at]);
            if (u) {
                out->local_norm+=(double)u[at]*u[at];
                out->local_delta+=((double)r[at]-u[at])*((double)r[at]-u[at]);
            }
        }
    }
    out->formula_tolerance=1e-6*(1+expected_max);
    out->incoming_norm=sqrt(out->incoming_norm); out->local_norm=sqrt(out->local_norm);
    out->returned_norm=sqrt(out->returned_norm); out->incoming_delta=sqrt(out->incoming_delta);
    out->local_delta=sqrt(out->local_delta);
    if (!isfinite(out->coordinate_error) || !isfinite(out->incoming_norm) ||
        !isfinite(out->local_norm) || !isfinite(out->returned_norm) ||
        !isfinite(out->incoming_delta) || !isfinite(out->local_delta) ||
        out->formula_error>out->formula_tolerance) {
        fprintf(stderr,"recall audit: slot %d %s formula error %.17g, tolerance %.17g\n",
            slot,names[kind],out->formula_error,out->formula_tolerance);
        return 0;
    }
    return 1;
}

int main(int argc,char **argv) {
    if (argc!=5) {
        fprintf(stderr,"usage: %s HOST INCOMING_EXACT_OR_PORTABLE LOCAL_EXACT_OR_PORTABLE RECALLED_PORTABLE\n",argv[0]);
        return 1;
    }
    int ok=0,matched=0;
    gguf_file *host=NULL,*files[3]={0};
    nt_experts *owners[3]={0};
    float *frame=NULL;
    measurement rows[SEATS*3];
    host=gguf_open(argv[1]);
    if (!host) goto done;
    for (int i=0; i<3; i++) {
        files[i]=gguf_open(argv[i+2]);
        if (!files[i] || !known_metadata(files[i])) goto done;
        owners[i]=nt_experts_open(host,argv[i+2],NULL);
        if (!owners[i]) {
            fprintf(stderr,"recall audit: cannot restore %s on exact HOST\n",argv[i+2]);
            goto done;
        }
        const gguf_kv *body=gguf_get_kv(files[i],"doe.host_sha256");
        if (!body || strcmp(body->val.str,owners[i]->host) || owners[i]->count ||
            owners[i]->observed || owners[i]->choices || !owners[i]->turns) goto done;
    }
    nt_experts *incoming=owners[0],*local=owners[1],*returned=owners[2];
    if (incoming->dim!=local->dim || incoming->layer!=local->layer ||
        incoming->depth!=local->depth || strcmp(incoming->host,local->host) ||
        !metadata_preserved(files[0],files[2]) || !state_preserved(incoming,returned)) goto done;
    const gguf_kv *version=gguf_get_kv(files[2],"doe.version");
    const gguf_kv *width=gguf_get_kv(files[2],"doe.width");
    int tensor=gguf_find_tensor(files[2],"doe.anchor_frame");
    if (!version || version->val.u32!=6 || !width || width->val.u32!=(unsigned)incoming->dim ||
        tensor<0 || files[2]->tensors[tensor].dtype!=GGUF_TYPE_F32 ||
        files[2]->tensors[tensor].ndim!=2 ||
        files[2]->tensors[tensor].shape[0]!=(uint64_t)incoming->dim ||
        files[2]->tensors[tensor].shape[1]!=NT_SPORE_ANCHORS) goto done;
    frame=gguf_dequant(files[2],tensor);
    if (!frame || !nt_spore_frame_valid(frame,incoming->dim)) goto done;
    for (int e=0; e<SEATS; e++) {
        nt_expert_identity id;
        if (nt_experts_seat_identity(incoming,e,&id)) goto done;
        int from=incoming->alive[e] ? nt_experts_find_identity(local,id.id) : -1;
        matched+=from>=0;
        for (int kind=0; kind<3; kind++)
            if (!measure(incoming,local,returned,frame,e,from,kind,rows+e*3+kind)) goto done;
    }
    puts("identity\tslot\tlocal_slot\tkind\tfloats\tcoordinates\tmatched\tincoming_l2\tlocal_l2\trecalled_l2\tincoming_delta_l2\tlocal_delta_l2\tnorm_change\tcoordinate_max_abs\torthogonal_max_abs\tformula_max_abs\tformula_tolerance\tunchanged");
    for (int i=0; i<SEATS*3; i++) {
        measurement *m=rows+i;
        printf("%s\t%d\t%d\t%s\t%d\t%d\t%d\t%.17g\t",m->id,m->slot,m->local_slot,
            m->kind,m->floats,m->coordinates,m->local_slot>=0,m->incoming_norm);
        if (m->local_slot>=0) printf("%.17g",m->local_norm); else fputs("na",stdout);
        printf("\t%.17g\t%.17g\t",m->returned_norm,m->incoming_delta);
        if (m->local_slot>=0) printf("%.17g",m->local_delta); else fputs("na",stdout);
        printf("\t%.17g\t%.17g\t",m->returned_norm-m->incoming_norm,m->coordinate_error);
        if (m->local_slot>=0) printf("%.17g",m->orthogonal_error); else fputs("na",stdout);
        printf("\t%.17g\t%.17g\t%d\n",m->formula_error,m->formula_tolerance,m->unchanged);
    }
    if (fflush(stdout) || ferror(stdout)) goto done;
    fprintf(stderr,"recall audit: matched=%d width=%d anchors=%d rows=%d state=unchanged formula=pass\n",
        matched,incoming->dim,NT_SPORE_ANCHORS,SEATS*3);
    ok=1;
done:
    if (!ok) fprintf(stderr,"recall audit: FAIL\n");
    free(frame);
    for (int i=0; i<3; i++) { nt_experts_free(owners[i]); gguf_close(files[i]); }
    gguf_close(host);
    return ok ? 0 : 1;
}
