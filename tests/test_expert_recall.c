/* The portable fixture's source frame is exactly axes 0..31. An independent
 * coordinate oracle can therefore distinguish incoming shared coefficients
 * from the selected local body's remaining coordinates without calling the
 * projector being tested. */
#include "harness/experts_internal.h"
#include "harness/archs.h"
#include "harness/spore_frame.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int checks, failures;
#define CHECK(c, what) do { checks++; if (!(c)) { \
    fprintf(stderr, "FAIL body recall: %s (line %d)\n", what, __LINE__); failures++; \
} } while (0)

typedef struct {
    gguf_file *g;
    const nt_arch *arch;
    void *model;
    bpe_tokenizer *tok;
    nt_expert_body body;
} body;

typedef struct {
    nt_experts value;
    nt_lineage_entry *ledger;
    float *weights;
} snapshot;

static body open_body(const char *path) {
    body b = {0};
    b.g = gguf_open(path);
    assert(b.g);
    b.arch = nt_pick_arch(b.g->arch);
    nt_dims dims;
    b.model = b.arch ? b.arch->load(b.g, &dims) : NULL;
    b.tok = bpe_load(path);
    assert(b.model && b.tok);
    b.body = (nt_expert_body){b.model, b.tok};
    return b;
}

static void close_body(body *b) {
    bpe_free(b->tok); b->arch->free(b->model); gguf_close(b->g);
}

static nt_experts *open_state(body *b, const char *path) {
    nt_experts *p = nt_experts_open_on_body(b->g, path, NULL, &b->body);
    assert(p);
    return p;
}

static snapshot remember(const nt_experts *p) {
    snapshot s = {0};
    memcpy(&s.value, p, sizeof(*p));
    size_t n = (size_t)SEATS * p->dim * RANK, v = (size_t)SEATS * p->dim;
    s.weights = malloc((2*n + v) * sizeof(float));
    s.ledger = malloc((size_t)p->ancestor_count * sizeof(*s.ledger));
    assert(s.weights && s.ledger);
    memcpy(s.weights, p->a, n*sizeof(float));
    memcpy(s.weights+n, p->b, n*sizeof(float));
    memcpy(s.weights+2*n, p->vote, v*sizeof(float));
    memcpy(s.ledger, p->ancestors, (size_t)p->ancestor_count*sizeof(*s.ledger));
    return s;
}

static int metadata_same(const nt_experts *p, const snapshot *s) {
    return !memcmp(p, &s->value, sizeof(*p)) &&
        !memcmp(p->ancestors, s->ledger, (size_t)p->ancestor_count*sizeof(*p->ancestors));
}

static int weights_same(const nt_experts *p, const snapshot *s) {
    size_t n = (size_t)SEATS * p->dim * RANK, v = (size_t)SEATS * p->dim;
    return !memcmp(p->a, s->weights, n*sizeof(float)) &&
        !memcmp(p->b, s->weights+n, n*sizeof(float)) &&
        !memcmp(p->vote, s->weights+2*n, v*sizeof(float));
}

static void forget(snapshot *s) { free(s->ledger); free(s->weights); }

static int same_file(const char *a, const char *b) {
    FILE *x = fopen(a, "rb"), *y = fopen(b, "rb");
    assert(x && y);
    int c, d, same = 1;
    do { c=fgetc(x); d=fgetc(y); if (c!=d) same=0; } while (c!=EOF && d!=EOF);
    same = same && !ferror(x) && !ferror(y);
    assert(!fclose(x) && !fclose(y));
    return same;
}

/* Native waking hooks and consolidation own the turn. The optional hit and
 * threshold interventions exercise birth/death without waiting for a chance
 * routing history, as in the lineage gate. */
static void turn(nt_experts *p, float value, int parent, int retire) {
    float *row = malloc((size_t)p->dim*sizeof(float)); assert(row);
    for (int t=0; t<3; t++) {
        for (int i=0; i<p->dim; i++) row[i] = value + (float)((i+3*t)%17)/32;
        assert(nt_experts_attend(p,p->layer,t,1,p->dim,row)==NT_OK);
        assert(nt_experts_observe(p,p->layer,t,1,p->dim,row)==NT_OK);
    }
    nt_experts_choice(p,1,1);
    if (parent>=0 || retire>=0) {
        for (int e=0; e<SEATS; e++) {
            p->age[e]=p->low[e]=0;
            if (p->alive[e]) p->vitality[e]=0.5f;
            p->hits[e]=0;
        }
        if (parent>=0) { p->age[parent]=21; p->vitality[parent]=0.99f; p->hits[parent]=p->count; }
        if (retire>=0) { p->vitality[retire]=0; p->low[retire]=7; }
    }
    assert(nt_experts_finish(p,1)==NT_OK);
    free(row);
}

static int matched_slot(const nt_experts *incoming, const nt_experts *local, int e) {
    if (!incoming->alive[e]) return -1;
    nt_expert_identity id;
    assert(nt_experts_seat_identity(incoming,e,&id)==NT_OK);
    return nt_experts_find_identity(local,id.id);
}

static void coordinate_oracle(const nt_experts *after, const snapshot *before,
                              const nt_experts *local, const snapshot *local_before) {
    int d=after->dim;
    assert(d==64 && local->dim==d);
    size_t n=(size_t)SEATS*d*RANK;
    const float *actual[]={after->a,after->b,after->vote};
    const float *old[]={before->weights,before->weights+n,before->weights+2*n};
    const float *saved[]={local_before->weights,local_before->weights+n,local_before->weights+2*n};
    const char *names[]={"analytic A coordinates", "analytic B coordinates", "analytic vote coordinates"};
    for (int field=0; field<3; field++) {
        int rank=field==2 ? 1 : RANK, equal=1;
        for (int e=0; e<SEATS; e++) {
            int from=matched_slot(after,local,e);
            for (int r=0; r<rank; r++) for (int x=0; x<d; x++) {
                size_t dst=(size_t)e*d*rank + (field==0 ? (size_t)x*rank+r : (size_t)r*d+x);
                float expected=old[field][dst];
                if (from>=0 && x>=NT_SPORE_ANCHORS) {
                    size_t src=(size_t)from*d*rank + (field==0 ? (size_t)x*rank+r : (size_t)r*d+x);
                    expected=saved[field][src];
                }
                if (actual[field][dst]!=expected) equal=0;
            }
        }
        CHECK(equal,names[field]);
    }
    CHECK(metadata_same(after,before),"incoming clocks, RNG, cursors, lifecycle, genealogy and frame remain exact");
    CHECK(metadata_same(local,local_before) && weights_same(local,local_before),"selected local state stays unchanged");
}

static void swap_seats(nt_experts *p, int left, int right) {
    size_t n=(size_t)p->dim*RANK;
    float *tmp=malloc(n*sizeof(float)); assert(tmp);
    float *fields[]={p->a,p->b,p->vote};
    for (int f=0; f<3; f++) {
        size_t size=f==2 ? (size_t)p->dim : n;
        memcpy(tmp,fields[f]+left*size,size*sizeof(float));
        memcpy(fields[f]+left*size,fields[f]+right*size,size*sizeof(float));
        memcpy(fields[f]+right*size,tmp,size*sizeof(float));
    }
    float *small[]={p->alive,p->vitality,p->frequency,p->cursor,p->age,p->low};
    for (size_t f=0; f<sizeof small/sizeof *small; f++) {
        float v=small[f][left]; small[f][left]=small[f][right]; small[f][right]=v;
    }
    int32_t id=p->seat_ancestor[left]; p->seat_ancestor[left]=p->seat_ancestor[right]; p->seat_ancestor[right]=id;
    free(tmp);
}

static void prepare_history(body *home, body *away) {
    nt_experts *local=open_state(home,NULL);
    turn(local,0.2f,-1,-1);
    /* Distinct body-local coordinates give the return a decisive oracle. */
    for (int e=0; e<FOUNDERS; e++) for (int x=NT_SPORE_ANCHORS; x<local->dim; x++) {
        local->vote[(size_t)e*local->dim+x]=0.03f*(e+1)+0.0001f*x;
        for (int r=0; r<RANK; r++) {
            local->a[((size_t)e*local->dim+x)*RANK+r]=0.02f*(e+1)+0.001f*r+0.0001f*x;
            local->b[(size_t)e*local->dim*RANK+(size_t)r*local->dim+x]=-0.025f*(e+1)-0.001f*r-0.0001f*x;
        }
    }
    assert(!nt_experts_save(local,"recall-local-exact") &&
           !nt_experts_save_portable(local,"recall-local-portable"));
    nt_experts *foreign=open_state(away,"recall-local-portable");
    turn(foreign,0.7f,-1,-1);
    assert(nt_experts_dream(foreign,2,-1,1)==NT_OK);
    int ids[]={32,33};
    nt_replay_episode *episode=nt_replay_capture(foreign,ids,NULL,2,NULL,0,NULL,0);
    assert(episode && nt_experts_replay(foreign,episode,2,1)==NT_OK);
    nt_replay_free(episode);
    assert(!nt_experts_save_portable(foreign,"recall-away"));
    nt_experts_free(foreign); nt_experts_free(local);
}

static void round_trip(body *home, body *away) {
    nt_experts *local=nt_experts_open(home->g,"recall-local-exact",NULL);
    nt_experts *incoming=open_state(home,"recall-away");
    assert(local && incoming->sleep.cycles==1 && incoming->replay.cycles==1);
    snapshot old=remember(incoming), saved=remember(local);
    int matched=-1;
    CHECK(nt_experts_recall_body(incoming,local,&matched)==NT_OK && matched==FOUNDERS,"return recalls four living identities");
    CHECK(!weights_same(incoming,&old),"return recovers learned coordinates outside the shared frame");
    coordinate_oracle(incoming,&old,local,&saved);
    assert(!nt_experts_save_portable(incoming,"recall-return"));
    snapshot recalled=remember(incoming);
    for (int i=0; i<256; i++) {
        matched=-1;
        CHECK(nt_experts_recall_body(incoming,local,&matched)==NT_OK && matched==FOUNDERS,
              "repeated recall resolves the same living addresses");
        CHECK(metadata_same(incoming,&recalled) && weights_same(incoming,&recalled),
              "repeated axis-frame recall preserves every state bit");
    }
    forget(&recalled); forget(&saved); forget(&old);
    nt_experts_free(incoming);
    char parent[64]="recall-return";
    for (int i=0; i<8; i++) {
        char out[64],back[64];
        snprintf(out,sizeof out,"recall-trip-%d-away",i);
        snprintf(back,sizeof back,"recall-trip-%d-home",i);
        nt_experts *foreign=open_state(away,parent);
        assert(!nt_experts_save_portable(foreign,out)); nt_experts_free(foreign);
        incoming=open_state(home,out);
        CHECK(nt_experts_recall_body(incoming,local,&matched)==NT_OK && matched==FOUNDERS,
              "restarted foreign round trip recalls by identity");
        assert(!nt_experts_save_portable(incoming,back));
        CHECK(same_file("recall-return",back),"eight untrained round trips retain identical home spores");
        nt_experts_free(incoming); snprintf(parent,sizeof parent,"%s",back);
    }
    nt_experts_free(local);
}

static void identities(body *home) {
    nt_experts *local=open_state(home,"recall-local-exact");
    nt_experts *incoming=open_state(home,"recall-away");
    swap_seats(local,0,1);
    snapshot before=remember(incoming), saved=remember(local);
    int matched=-1;
    CHECK(nt_experts_recall_body(incoming,local,&matched)==NT_OK && matched==FOUNDERS,
          "relocated local seats resolve by birth address");
    coordinate_oracle(incoming,&before,local,&saved);
    forget(&saved); forget(&before); nt_experts_free(incoming); nt_experts_free(local);

    local=open_state(home,"recall-local-exact"); incoming=open_state(home,"recall-away");
    turn(incoming,0.4f,0,-1);   /* A new incoming child in seat four. */
    turn(incoming,0.5f,-1,2);   /* Incoming identity two is retired. */
    turn(local,0.9f,3,1);      /* A different local child; identity one is retired. */
    CHECK(incoming->alive[4] && local->alive[4] &&
          incoming->seat_ancestor[4] && local->seat_ancestor[4],"both histories occupy the reused numeric seat");
    before=remember(incoming); saved=remember(local);
    CHECK(nt_experts_recall_body(incoming,local,&matched)==NT_OK && matched==2,
          "new and retired identities do not receive another birth's body memory");
    coordinate_oracle(incoming,&before,local,&saved);
    forget(&saved); forget(&before); nt_experts_free(incoming); nt_experts_free(local);
}

static void noops_and_refusals(body *home, body *away, body *aligned) {
    nt_experts *local=nt_experts_open(home->g,"recall-local-exact",NULL);
    assert(local && !local->frame);
    snapshot saved=remember(local);
    int matched=99;
    CHECK(nt_experts_recall_body(local,local,&matched)==NT_OK && matched==FOUNDERS &&
          metadata_same(local,&saved) && weights_same(local,&saved),"self recall is exact without calibration or decoder");
    CHECK(nt_experts_recall_body(local,local,NULL)==NT_OK,"matched output is optional");
    forget(&saved);
    nt_experts *other=open_state(aligned,NULL);
    turn(other,0.3f,-1,-1);
    assert(!nt_experts_save_portable(other,"recall-independent")); nt_experts_free(other);
    nt_experts *incoming=open_state(home,"recall-independent");
    snapshot before=remember(incoming); saved=remember(local);
    CHECK(nt_experts_recall_body(incoming,local,&matched)==NT_OK && matched==0 &&
          metadata_same(incoming,&before) && weights_same(incoming,&before),"unrelated birth addresses are an exact no-op");
    CHECK(metadata_same(local,&saved) && weights_same(local,&saved),"no-match donor remains unchanged");
    forget(&before); forget(&saved); nt_experts_free(incoming);

    incoming=open_state(home,"recall-away");
    nt_experts *foreign=open_state(away,"recall-away");
    before=remember(incoming);
    CHECK(nt_experts_recall_body(incoming,foreign,&matched)==NT_E_STATE && matched==0 &&
          metadata_same(incoming,&before) && weights_same(incoming,&before),"a different host is refused atomically");
    nt_experts_free(foreign);
    foreign=open_state(aligned,"recall-independent");
    CHECK(incoming->dim==foreign->dim && incoming->layer==foreign->layer &&
          nt_experts_recall_body(incoming,foreign,&matched)==NT_E_STATE && matched==0 &&
          metadata_same(incoming,&before) && weights_same(incoming,&before),
          "same-shaped different host bytes are refused before identity matching");
    nt_experts_free(foreign);
    CHECK(nt_experts_recall_body(NULL,local,&matched)==NT_E_ARG && matched==0 &&
          nt_experts_recall_body(incoming,NULL,NULL)==NT_E_ARG,"null owners are refused");
    int *changed[]={&local->dim,&local->layer,&local->count,&local->observed,&local->choices,
                    &incoming->count,&incoming->observed,&incoming->choices};
    for (size_t i=0; i<sizeof changed/sizeof *changed; i++) {
        int old=*changed[i]; *changed[i]=old+1;
        matched=99;
        CHECK(nt_experts_recall_body(incoming,local,&matched)==NT_E_STATE && matched==0 &&
              weights_same(incoming,&before),"shape or unfinished boundary is refused before writing");
        *changed[i]=old;
    }
    uint64_t old_turns=local->turns; local->turns=0;
    CHECK(nt_experts_recall_body(incoming,local,&matched)==NT_E_STATE && matched==0,
          "uncompleted local founder state is refused");
    local->turns=old_turns;
    old_turns=incoming->turns; incoming->turns=0;
    CHECK(nt_experts_recall_body(incoming,local,&matched)==NT_E_STATE && matched==0,
          "uncompleted incoming founder state is refused");
    incoming->turns=old_turns;
    CHECK(metadata_same(incoming,&before) && weights_same(incoming,&before),"all refusals preserve the incoming state");
    /* Fail only after three earlier matching experts were staged. */
    size_t at=(size_t)(FOUNDERS-1)*local->dim*RANK+(size_t)(RANK-1)*local->dim+local->dim-1;
    float old=local->b[at]; local->b[at]=NAN;
    CHECK(nt_experts_recall_body(incoming,local,&matched)==NT_E_STATE && matched==0 &&
          metadata_same(incoming,&before) && weights_same(incoming,&before),"late numerical rejection rolls back every staged expert");
    local->b[at]=old;
    forget(&before); nt_experts_free(incoming); nt_experts_free(local);

    incoming=nt_experts_open(home->g,"recall-local-exact",NULL);
    local=nt_experts_open(home->g,"recall-local-exact",NULL);
    assert(incoming && local);
    before=remember(incoming);
    CHECK(nt_experts_recall_body(incoming,local,&matched)==NT_E_STATE && matched==0 &&
          metadata_same(incoming,&before) && weights_same(incoming,&before),
          "matching owners without a calibratable body are refused unchanged");
    forget(&before); nt_experts_free(incoming); nt_experts_free(local);
}

int main(void) {
    body home=open_body("source"), away=open_body("wider"), aligned=open_body("aligned");
    prepare_history(&home,&away);
    round_trip(&home,&away);
    identities(&home);
    noops_and_refusals(&home,&away,&aligned);
    close_body(&aligned); close_body(&away); close_body(&home);
    printf("Expert body recall: %d checks, %d failures\n",checks,failures);
    return failures ? 1 : 0;
}
