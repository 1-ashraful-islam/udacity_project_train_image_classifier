"""
Predict flower name from an image with `predict.py` along with the probability of that name. That is, you'll pass in a single image `/path/to/image` and return the flower name and class probability.

* Basic usage: `python predict.py /path/to/image checkpoint`
* Options: 
    * Return top K most likely classes: `python predict.py input checkpoint --top_k 3` 
    * Use a mapping of categories to real names: `python predict.py input checkpoint --category_names cat_to_name.json` 
    * Use GPU for inference: `python predict.py input checkpoint --gpu`
"""
import torch
import helper
import json

def main():
    # define command line argument parsing
    
    # parse command line arguments
    in_args = helper.get_predict_input_args()

    # load label mapping from json file
    cat_to_name = None
    if in_args.category_names:
        with open(in_args.category_names, 'r') as f:
            cat_to_name = json.load(f)

    
    #load the checkpoint and rebuild the model
    model, _, _ = helper.load_checkpoint(in_args.checkpoint)

    # process the image for the model inference
    image_input = helper.process_image(in_args.image_path)

    # select the device to use for inference (select cpu by default since its a one shot inference)
    device = helper.get_device(in_args.gpu)
    

    # add batch of size 1 to image
    image = image_input.unsqueeze(0)

    # move the image to the device (GPU or CPU)
    image = image.to(device)
    # move the model to the device (GPU or CPU)
    model = model.to(device)

    # evaluate the image using the model
    model.eval()

    # invert the class_to_idx dictionary
    idx_to_class = {v: k for k, v in model.extras.get('class_to_idx').items()}
    
    
    with torch.no_grad():
        
        ps = torch.exp(model.forward(image))
        top_ps, top_class = ps.topk(in_args.top_k, dim=1)

        # convert the tensors to numpy arrays
        top_ps = top_ps[0].numpy()
        top_class = top_class[0].numpy()
        top_ps_list = top_ps.tolist()
        # use the classes to get the class keys
        top_class_keys = [idx_to_class[i.item()] for i in top_class]


    #print the results
    print("Top {} classes:".format(in_args.top_k))
    for i in range(in_args.top_k):
        if cat_to_name:
            print("Prediction: {}, Class: {}, Probability: {:.3f}".format(cat_to_name[top_class_keys[i]], top_class_keys[i], top_ps_list[i]))
        else:
            print("Class: {}, Probability: {:.3f}".format(top_class_keys[i], top_ps_list[i]))



#when the program is run from the command line
if __name__ == "__main__":
    main()