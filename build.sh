ljspath=$1

ln -s $ljspath DUMMY
git submodule init
git submodule update
cd monotonic_align
python setup.py build_ext --inplace