mkdir -p data/train/fire && rm -f data/train/fire/*
mkdir -p data/train/nofire && rm -f data/train/nofire/*
mkdir -p data/validate/fire && rm -f data/validate/fire/*
mkdir -p data/validate/nofire && rm -f data/validate/nofire/*
mkdir -p data/test/fire && rm -f data/test/fire/*
mkdir -p data/test/nofire && rm -f data/test/nofire/*

# Copy datasets into appropriate location
training_sets="corsican deepquest"
validation_sets="cair"
testing_sets="phylake"

for dset in $training_sets; do
  cp /storage/deepfire/data/$dset/fire/* data/train/fire
  cp /storage/deepfire/data/$dset/nofire/* data/train/nofire
done

for dset in $validation_sets; do
  cp /storage/deepfire/data/$dset/fire/* data/validate/fire
  cp /storage/deepfire/data/$dset/nofire/* data/validate/nofire
done

for dset in $testing_sets; do
  cp /storage/deepfire/data/$dset/fire/* data/test/fire
  cp /storage/deepfire/data/$dset/nofire/* data/test/nofire
done
