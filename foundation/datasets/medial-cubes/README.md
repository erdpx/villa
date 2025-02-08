Development branch...

Some scripts are heavily edited from james'darby code

conda env create -f environment.yml
conda activate vol_isntance_to_mesh
pip install -r requirement.txt

git clone https://github.com/giorgioangel/scikit-image.git

cd scikit-image

git switch medial-surface-thinning

pip install build wheel
python -m build --wheel

pip install dist/scikit_image-0.24.1rc0.dev0-cp312-cp312-linux_x86_64.whl

cd ..

download the harmonized-cubes.7z dataset

run label_thinner.py on harmonized-cubes with --no-mask-out --morph 20

run final_labeler.py on harmonized-cubes, destination: some new folder, pattern _thin_clean
