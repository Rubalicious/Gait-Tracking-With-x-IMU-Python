$scalings=(1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5)
$taus=(.01, .05, .1, .15, .2)

foreach($scaling in $scalings){
    foreach($tau in $taus){
        python ./script.py --scaling $scaling --tau $tau
    }
}