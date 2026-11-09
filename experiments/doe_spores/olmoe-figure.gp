# gnuplot -c olmoe-figure.gp DIRECTORY_WITH_AML_TABLES olmoe-routing.svg
set datafile separator "\t"
array pair[3] = ["q0-plain/q0-awake", "q0-awake/q0-dream", "q0-plain/km-plain"]
array changes[3]
array mass_l1[3]
set print $curves
do for [layer=0:15] {
    do for [i=1:3] {
        file = ARG1 . (i == 3 ? "/quantization.tsv" : "/routes.tsv")
        stats file using (strcol(1) eq pair[i] && $3 == layer ? $4 : 1/0) nooutput
        rows = STATS_sum
        stats file using (strcol(1) eq pair[i] && $3 == layer ? $5 : 1/0) nooutput
        changes[i] = STATS_sum
        stats file using (strcol(1) eq pair[i] && $3 == layer ? $4*$9 : 1/0) nooutput
        mass_l1[i] = STATS_sum/rows
    }
    print sprintf("%d\t%.0f\t%.0f\t%.0f\t%.9g\t%.9g\t%.9g", layer, changes[1],changes[2],changes[3],mass_l1[1],mass_l1[2],mass_l1[3])
}
set print
set terminal svg size 1080,690 fixed enhanced font "DejaVu Sans,16" background rgb "white"
set output ARG2
set multiplot layout 2,1 margins 0.13,0.975,0.12,0.83 spacing 0.12,0.075
set xrange [-0.2:15.2]
set xtics 1 nomirror textcolor rgb "#48525c"
set ytics nomirror textcolor rgb "#48525c"
set border 3 lc rgb "#a8afb7"
set grid ytics back lc rgb "#e1e5e9"
set style line 1 lc rgb "#3b65a2" pt 7 ps 0.8 lw 2.4
set style line 2 lc rgb "#a64426" pt 5 ps 0.8 lw 2.4
set style line 3 lc rgb "#087c6e" pt 9 ps 0.9 lw 2.5 dt 2
set arrow 1 from 8,graph 0 to 8,graph 1 nohead dt 3 lw 1.0 lc rgb "#69727e" back
set label 1 "OLMoE native FFN routing on the same 18 input positions" at screen 0.095,0.960 font ",19" front
set key at screen 0.095,0.910 left top horizontal Left reverse samplen 2 spacing 1 maxrows 1 font ",15"
set ylabel "Changed top-8 sets\n(out of 18)"
set yrange [-0.4:19]
set ytics 0,3,18
set format x ""
plot $curves using 1:2 with linespoints ls 1 title "Seat the Gemma spore", \
     '' using 1:3 with linespoints ls 2 title "Record one dream", \
     '' using 1:4 with linespoints ls 3 title "Switch quantization (plain)"
unset label 1
unset key
set xlabel "Decoder block    (parliament at block 8)"
set ylabel "Mean L1 distance\nbetween applied weights"
# Lower tick is reserved for measured exact zeros. The positive region is logarithmic.
zero = 1e-9
set logscale y
set yrange [zero:1]
set ytics ("0" zero, "10^{-7}" 1e-7, "10^{-5}" 1e-5, "10^{-3}" 1e-3, "10^{-1}" 1e-1, "1" 1)
set format x "%g"
plot $curves using 1:($5==0 ? zero : $5) with linespoints ls 1, '' using 1:($6==0 ? zero : $6) with linespoints ls 2, '' using 1:($7==0 ? zero : $7) with linespoints ls 3
unset multiplot
