# Compare the selected top-8 on two frozen prompt prefixes. expected is the
# receiving tokenizer's comma-separated ID sequence. Decode is kept in the
# full traces; its generated inputs may differ between conditions.
BEGIN {
    OFS = "\t"
    n = split(expected, input, ",")
    if (n < 1 || expected !~ /^[0-9]+(,[0-9]+)*$/ || ARGC != 3) fail("need two traces and expected input IDs")
    if (scope == "") scope = "parliament"
    if (scope != "parliament" && scope != "quantization" && scope != "factor-replay") fail("unknown comparison scope")
    if (scope == "quantization" && source_sha != "-" &&
        (length(source_sha) != 64 || source_sha !~ /^[0-9a-f]+$/)) fail("missing common source archive SHA-256")
}
function fail(message) {
    print "olmoe_routes: " message > "/dev/stderr"
    failed = 1
    exit 1
}
function value(field, name) {
    if (index(field,name "=") != 1) fail("missing " name)
    return substr(field,length(name)+2)
}
/^doe choice step=0 / {
    if (++choices[FILENAME] > 1) fail("multiple generations in " FILENAME)
}
/^moe route / && !choices[FILENAME] {
    if (NF != 7) fail("unexpected route columns")
    side = FILENAME == ARGV[1] ? 1 : 2
    layer = value($3,"layer"); pos = value($4,"pos"); token = value($5,"token")
    if (layer !~ /^[0-9]+$/ || pos !~ /^[0-9]+$/ || token !~ /^[0-9]+$/) fail("invalid route coordinates")
    layer += 0; pos += 0
    if (layer >= 16 || pos >= n || token != input[pos+1]) fail("route differs from the pinned prompt")
    key = side SUBSEP layer SUBSEP pos
    if (++seen[key] != 1) fail("duplicate prompt route")
    ids[key] = value($6,"ids"); weights[key] = value($7,"weights")
    if (split(ids[key],id,",") != 8 || split(weights[key],w,",") != 8) fail("expected top-8")
    sum = 0
    for (k = 1; k <= 8; k++) {
        if (id[k] !~ /^[0-9]+$/ || id[k]+0 >= 64) fail("invalid OLMoE bank ID")
        if (w[k] !~ /^[0-9]+([.][0-9]+)?([eE][+-]?[0-9]+)?$/ || w[k]+0 > 1) fail("invalid route weight")
        if (k > 1 && w[k]+0 > w[k-1]+0) fail("top-8 weights out of order")
        bank = key SUBSEP (id[k]+0)
        if (bank in selected) fail("duplicate OLMoE bank")
        selected[bank] = w[k]+0
        sum += w[k]
    }
    if (sum < 0.125-0.000001 || sum > 1.000001) fail("invalid top-8 mass on a 64-bank softmax")
    mass[key] = sum
}
END {
    if (failed) exit 1
    if (choices[ARGV[1]] != 1 || choices[ARGV[2]] != 1) fail("missing first choice")
    for (layer = 0; layer < 16; layer++) {
        sets = ranks = replaced = top1 = l1sum = l1max = before = after = 0
        first = -1
        for (pos = 0; pos < n; pos++) {
            a = 1 SUBSEP layer SUBSEP pos; b = 2 SUBSEP layer SUBSEP pos
            if (!(a in seen) || !(b in seen)) fail("missing prompt route")
            if (scope == "factor-replay") {
                if (ids[a] != ids[b] || weights[a] != weights[b]) fail("factor replay differs at layer " layer ", pos " pos)
                continue
            }
            if (scope == "parliament" && layer < 8 && (ids[a] != ids[b] || weights[a] != weights[b]))
                fail("route changed before layer 8")
            split(ids[a],aid,","); split(ids[b],bid,",")
            drop = l1 = 0
            for (k = 1; k <= 8; k++) {
                old = a SUBSEP aid[k]; new = b SUBSEP aid[k]
                if (new in selected) {
                    delta = selected[old] - selected[new]
                    l1 += delta < 0 ? -delta : delta
                } else {
                    drop++
                    l1 += selected[old]
                }
                if (!((a SUBSEP bid[k]) in selected)) l1 += selected[b SUBSEP bid[k]]
            }
            if (drop) {
                sets++
                if (first < 0) first = pos
            } else if (ids[a] != ids[b]) ranks++
            replaced += drop
            top1 += aid[1] != bid[1]
            l1sum += l1
            if (l1 > l1max) l1max = l1
            before += mass[a]; after += mass[b]
        }
        if (scope == "factor-replay") continue
        printf "%s\t%s\t%d\t%d\t%d\t%d\t%d\t%d\t%.9g\t%.9g\t%.9g\t%.9g\t%d", \
               pair,probe,layer,n,sets,ranks,replaced,top1,l1sum/n,l1max,before/n,after/n,first
        if (scope == "quantization") printf "\t%s", source_sha
        printf "\n"
    }
    if (scope == "factor-replay") print 16*n
}
