# Compare matched prefill elections, or rebuild a greedy prefix for recurrent replay.
BEGIN {
    OFS = "\t"
    if (mode !~ /^(choices|prefix|cached|votes)$/) fail("unknown mode")
    if (ARGC != ((mode == "cached" || mode == "votes") ? 3 : 2)) fail("wrong number of traces")
    if (mode == "prefix" || mode == "cached") {
        if (expected !~ /^[0-9]+(,[0-9]+)*$/) fail("missing prompt token IDs")
        initial = split(expected, prompt_ids, ",")
    }
}
function fail(reason) { print "lfm3_trace: " reason > "/dev/stderr"; failed = 1; exit 1 }
function value(s, key) {
    if (index(s,key "=") != 1) fail("missing " key)
    return substr(s,length(key)+2)
}
function real(x) { return x ~ /^-?[0-9]+([.][0-9]+)?([eE][+-]?[0-9]+)?$/ }
function unit(x) { return real(x) && x+0 >= 0 && x+0 <= 1 }
function absval(x) { return x < 0 ? -x : x }
{ side = FILENAME == ARGV[1] ? 1 : 2 }
mode == "votes" && /^probe=/ {
    probe = value($1,"probe"); count = value($2,"tokens")
    if (NF != 2 || probe !~ /^[01]$/ || count !~ /^[1-9][0-9]*$/ || ((side SUBSEP probe) in size))
        fail("invalid probe header")
    seq = side SUBSEP probe
    size[seq] = count+0
}
mode == "votes" && /^doe vote / {
    pos = value($3,"pos"); k = value($4,"k"); c = value($5,"consensus"); gates = value($6,"gates")
    if (NF != 6 || !(seq in size) || pos !~ /^[0-9]+$/ || pos+0 >= size[seq] || k !~ /^[2-6]$/ ||
        !unit(c) || split(gates,v,",") != 6) fail("invalid vote coordinates")
    sum = 0
    for (i = 1; i <= 6; i++) {
        if (!unit(v[i])) fail("invalid living gate")
        sum += v[i]
    }
    if (sum < .999999 || sum > 1.000001 || ((seq SUBSEP pos) in votes)) fail("missing mass or duplicate vote")
    votes[seq,pos] = k OFS c OFS gates
}
mode != "votes" && /^doe choice / {
    step = value($3,"step"); id = value($4,"id"); l = value($5,"logit")
    if (NF != 7 || step !~ /^[0-9]+$/ || step+0 != n[side]+0 || id !~ /^[0-9]+$/ ||
        !real(l) || value($6,"best") != id || value($7,"best_logit") != l) fail("invalid greedy choice")
    ids[side,step] = id; logits[side,step] = l; n[side]++
    if (mode == "choices") print label,step,id,l
}
END {
    if (failed) exit 1
    if (mode == "votes") {
        for (p = 0; p < 2; p++) {
            a = 1 SUBSEP p; b = 2 SUBSEP p
            if (!(a in size) || !(b in size) || size[a] != size[b]) fail("matched probe sizes differ")
            for (j = 0; j < size[a]; j++) {
                x = a SUBSEP j; y = b SUBSEP j
                if (!(x in votes) || !(y in votes) || votes[x] != votes[y]) fail("living vote changed at probe " p ", pos " j)
            }
            print label,"P" p,size[a]
        }
        exit 0
    }
    if (!n[1] || n[1] > 8) fail("empty or overlong greedy trace")
    if (mode == "choices") exit 0
    # Feed at most three generated IDs, leaving a following observed choice to compare.
    step = n[1] > 3 ? 3 : n[1]-1
    if (step == 0) fail("no incremental choice to replay")
    if (mode == "prefix") {
        csv = expected
        for (i = 0; i < step; i++) csv = csv "," ids[1,i]
        print csv
    } else {
        if (n[2] != 1 || ids[1,step] != ids[2,0]) fail("fresh prefill greedy ID differs")
        error = absval(logits[1,step]-logits[2,0])
        if (error > 0.001+0.0001*absval(logits[1,step])) fail("cached logit differs from fresh prefill")
        print label,initial,step,ids[1,step],logits[1,step],logits[2,0],error
    }
}
