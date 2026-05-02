Not actual md, but just for notes

## Setup
```baash
conda create -n clipenv python=3.10
conda activate clipenv
pip install -r requirements.txt
```

Then the running path looks like 

  1. Export all models to ONNX                                                                                                                                
  ```bash
  python src/export_onnx.py --all
  ```
  2. Upload the dataset to QAI Hub                                                                                                                            
```bash
python src/upload_dataset.py --model <model>
```    
   Right now this uploads 50 things from the coco dataset, idk bro it's a start 
                                                                                                                                                  
  3. Compile, profile, and run inference                                                                                                                      
```bash
python src/compile_and_profile.py \
    --image-dataset-id <img_id> \                                                                                                                             
    --text-dataset-id <txt_id>                              
    --model <model> 
```
  Optionally: set --profile to generate profile runs of our models after inference  

For each of the five models this compiles all encoders, fires off profile jobs, runs inference, and prints Recall@10. Add --topk faiss to use the baked-index  
  variant instead. Compile job IDs are printed as they're submitted — save these if you want to re-run inference later without recompiling.
                                                                                                                                                              
  4. (Optional) Re-run inference on already-compiled models 
  ```bash
python src/inference.py \                                                                                                                                   
    --img-compiled-id <id> \                                
    --txt-compiled-id <id> \                                                                                                                                  
    --topk-compiled-id <id> \
    --image-dataset-id <img_id> \                                                                                                                             
    --text-dataset-id <txt_id>                              
```