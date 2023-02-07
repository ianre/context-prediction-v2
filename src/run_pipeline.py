import os, sys
from pipeline_scripts.mask_context_evaluation import Context_Iterator
from pipeline_scripts.contour_extraction import Contour_Iterator
from pipeline_scripts.metrics import Metrics_Iterator
import pipeline_scripts.utils
import time

def getTrialFrames(CWD,TASK,TRIAL):
    TrialRoot = os.path.join(CWD,"data","images",TRIAL)
    for root, dirs, files in os.walk(TrialRoot):
        Frames = files
        Frames = [f for f in files if ".png" in f]
        break    
    frameNumbers = []
    for f in Frames:
        frameNumber = f.replace(".png","").split("_")[1]
        frameNumbers.append(frameNumber)
    return frameNumbers

def all_trial_pipeline(MASK_SET,TRIALS,TASK,CWD):
    # for loop over all of the TRIALS in a TASK
    for TRIAL in TRIALS:
        TRIAL_FRAMES = getTrialFrames(CWD,TASK,TRIAL) # formatted as the XXXX in 'frame_XXXX.png'
        I = Contour_Iterator(MASK_SET,TRIAL,CWD)
        label_classes, ContourFiles, RingFile = I.ExtractContoursTask(TRIAL,TRIAL_FRAMES)

    return
    
    TRIAL = trials[0]
    TRIAL_FRAMES = getTrialFrames(CWD,TASK,TRIAL) # formatted as the XXXX in 'frame_XXXX.png'
    TOTAL_TIME = 0
    TOTAL_EPOCH = 0
    
    
    print("Pipeline Running for task ",TASK,"trial",TRIAL)
    while(EPOCH * BATCH_SIZE < len(TRIAL_FRAMES)):
        print("EPOCH ", EPOCH, ":", TRIAL_FRAMES[BATCH_SIZE*EPOCH:BATCH_SIZE*EPOCH+BATCH_SIZE])
        start_time = time.time()
        I = Contour_Iterator(TASK,CWD)
        # Format: ["2023_grasper_L_masks","other folders"], ["2023_grasper_L", "label names"], ["C://...//2023_grasper_L_masks/Knot_Tying_S03_T02_0.json", "other filenames"] 
        label_classes, label_classNames, ContourFiles, RingFile = I.ExtractContours(BATCH_SIZE,EPOCH,TRIAL,TRIAL_FRAMES[BATCH_SIZE*EPOCH:BATCH_SIZE*EPOCH+BATCH_SIZE])
        I = Context_Iterator(TASK,CWD)
        I.GenerateContext(BATCH_SIZE,EPOCH,TRIAL,TRIAL_FRAMES[BATCH_SIZE*EPOCH:BATCH_SIZE*EPOCH+BATCH_SIZE],label_classes, label_classNames,ContourFiles,RingFile,SAVE=True)
        end_time = time.time()
        epoch_time = end_time - start_time
        if((EPOCH+1) * BATCH_SIZE < len(TRIAL_FRAMES)):
            TOTAL_TIME += epoch_time
            TOTAL_EPOCH +=1
        print("   %s s \n" % round(epoch_time,3), end="")
        EPOCH +=1

    print("\n---------------------- Time Analysis --------------------")
    print("Average runtime for batch size=",BATCH_SIZE," is ", round(TOTAL_TIME/TOTAL_EPOCH,4),"seconds")
    print("                                        ", round(TOTAL_TIME/TOTAL_EPOCH,6)*1000,"ms")
    print("Total Execution time",round(TOTAL_TIME,5),"seconds")
    print("\n---------------------- Metrics --------------------")
    I = Metrics_Iterator(TASK,CWD)
    I.IOU()
    quit();


def run_batched_pipeline(trials, TASK, BATCH_SIZE, CWD):  
    BATCH_SIZE = 25
    EPOCH = 0
    TRIAL = trials[0]
    TRIAL_FRAMES = getTrialFrames(CWD,TASK,TRIAL) # formatted as the XXXX in 'frame_XXXX.png'
    TOTAL_TIME = 0
    TOTAL_EPOCH = 0
    
    
    print("Pipeline Running for task ",TASK,"trial",TRIAL)
    while(EPOCH * BATCH_SIZE < len(TRIAL_FRAMES)):
        print("EPOCH ", EPOCH, ":", TRIAL_FRAMES[BATCH_SIZE*EPOCH:BATCH_SIZE*EPOCH+BATCH_SIZE])
        start_time = time.time()
        I = Contour_Iterator(TASK,CWD)
        # Format: ["2023_grasper_L_masks","other folders"], ["2023_grasper_L", "label names"], ["C://...//2023_grasper_L_masks/Knot_Tying_S03_T02_0.json", "other filenames"] 
        label_classes, label_classNames, ContourFiles, RingFile = I.ExtractContours(BATCH_SIZE,EPOCH,TRIAL,TRIAL_FRAMES[BATCH_SIZE*EPOCH:BATCH_SIZE*EPOCH+BATCH_SIZE])
        I = Context_Iterator(TASK,CWD)
        I.GenerateContext(BATCH_SIZE,EPOCH,TRIAL,TRIAL_FRAMES[BATCH_SIZE*EPOCH:BATCH_SIZE*EPOCH+BATCH_SIZE],label_classes, label_classNames,ContourFiles,RingFile,SAVE=True)
        end_time = time.time()
        epoch_time = end_time - start_time
        if((EPOCH+1) * BATCH_SIZE < len(TRIAL_FRAMES)):
            TOTAL_TIME += epoch_time
            TOTAL_EPOCH +=1
        print("   %s s \n" % round(epoch_time,3), end="")
        EPOCH +=1

    print("\n---------------------- Time Analysis --------------------")
    print("Average runtime for batch size=",BATCH_SIZE," is ", round(TOTAL_TIME/TOTAL_EPOCH,4),"seconds")
    print("                                        ", round(TOTAL_TIME/TOTAL_EPOCH,6)*1000,"ms")
    print("Total Execution time",round(TOTAL_TIME,5),"seconds")
    print("\n---------------------- Metrics --------------------")
    I = Metrics_Iterator(TASK,CWD)
    I.IOU()
    quit();

def main():    
    CWD=os.getcwd()

    # DEFAULTS
    TASK = "Knot_Tying"
    BATCH_or_ALL = "ALL"
    MASK_SET="2023_ICRA"
    try:
        TASK=sys.argv[1]
        BATCH_or_ALL=sys.argv[2]
        MASK_SET=sys.argv[3]
    except:
        print("WARNING: no TASK BATCH, or MASK SET provided","\n Usage: python run_pipeline.py <TASK: Knot_Tying, Suturing> <batch size|all> <MASK_SET: 2023_DL, 2023_ICRA,...>","Default Task "+TASK + " for ALL trials")
    taskDir = os.path.join(CWD,"data","images") # Images dictate the iteration frames
    TRIALS = [] 
    for root, dirs, files in os.walk(taskDir):
        TRIALS = [x for x in dirs if TASK in x]
        break

    print("Trials:",TRIALS)

    if(BATCH_or_ALL == "ALL"):
        all_trial_pipeline(MASK_SET,TRIALS,TASK,CWD);
    else:
        BATCH_SIZE = int(BATCH_or_ALL)
        run_batched_pipeline(TRIALS,TASK,BATCH_SIZE,CWD);

main();
