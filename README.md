## DL_Competition

# How to run SimVP

Setup environment:
```
cd ./our_OpenSTL
conda env create -f environment.yml
conda activate OpenSTL  # Or check if your env already has the requirements.
python setup.py develop # This will create 'openstl' as a package
```

Command to run : 

```
cd ./our_OpenSTL
python tools/clevrer_train.py \
    --epoch 100 \
    --dataname "clevrer" \
    --data_root "../../../Dataset_Student" \
    --method "SimVP" \
    --ex_name "14000cleanvids_simvp_batch" \
    --auto_resume \
    --batch_size 1 \
    --val_batch_size 4 \
    --fp16 \
    --device "cuda" \
    --use_gpu True
```

--dataname  : to load our data

--data_root : Where Dataset_Student lives

--ex_name   : Name of the experiment. As of now, the model files will be saved in ./work_dirs/exp_name/