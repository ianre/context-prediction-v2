# Context Prediction ~~V1~~ V2

* `src/run_pipeline.py` is the script generates context labels based on the deeplab instrument masks without kinematics
* Data is in the same format as the DSA_Thread_Sampling repo

# Naming Conventions 
Tasks can be Needle_Passing, Knot_Tying, Suturing

Masks belong to sets such as 2023_ICRA, COGITO_GT, 2023_DL, ...

Each task subject trial combination appears under each mask folder as  ```<Task>_S<Subject number>_T<Trial number>```

## Folder Structure
* data
    * context_labels
        * consensus
        * surgeon
        * `<Labeler>`
    * contours
    * masks
        * 2023_ICRA
        * COGITO_GT
        * 2023_DL
            * leftgrasper
            * needle
            * rightgrasper
            * ring
            * thread
                * ```<Task>_<Subject>_<Trial>```
                    * frame_0001.png
    * images
        * ```<Task>_<Subject>_<Trial>```
            * frame_0001.png
* eval 
    * labeled_images
        * ```<Task>_<Subject>_<Trial>```
            * frame_0001.png
    * pred_context_labels
        * 2023_ICRA
        * COGITO_GT
        * 2023_DL           
            * ```<Task>_<Subject>_<Trial>```.txt
* src
    * run_pipeline.py -- runs entire context prediction pipeline

# Scripts

## run_pipeline.py

* JSONInterface_cogito: Helps to extract polygons, keypoints, and polylines from cogito Annotaiton JSON files
* JSONInterface_via: Helps to extract polygons, keypoints, and polylines from VGG Image Annotator (VIA) Annotaiton JSON files
* Iterator: loops through all images and generates context labels

## Notes:

install requirements.txt

run `python run_pipeline.py <Task name>`

task name can be one of:
- Knot_Tying
- Needle_Passing
- Suturing


