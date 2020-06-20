# python src/plot.py "data/base_wd"
# TODO: skip mechanism

for d1 in ./data/*; do
    for d2 in "$d1"/*; do
        if [ -d "$d2" ]; then
            python src/plot.py "$d2"
        fi
    done
    if [ -d "$d1" ]; then
        python src/plot.py "$d1"
    fi
done
