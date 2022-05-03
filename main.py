import os
import subprocess
import psutil
import mysql.connector
from flask import Flask

app = Flask(__name__)

BASE_PATH = os.getcwd()
TRAIN_PATH = os.path.join(BASE_PATH, "auto_train.py")
FILE_STORAGE_PATH = "Z:\\private\\training\\original"
train_proc = subprocess.Popen(['echo', 'Server Start!'], shell=True)
train_proc.terminate()


try:
    db = mysql.connector.connect(host='172.25.0.13', port=3306, database='kesco', user='kesco', password='kesco_user')
except:
    print("DB Connecting Failure")
    raise


def on_terminate(proc):
    print("process {} terminated with exit code {}".format(proc, proc.returncode))

def load_txt_arr(txt_path):
    f = open(txt_path, 'r')
    lines = f.readlines()
    return lines

def load_json_arr(json_path):
    lines = []
    with open(json_path, 'r') as f:
        for line in f:
            lines.append(json.loads(line))
    return lines

def check_session(db, session_id, weight_id=None):
    cursor = db.cursor()
    sql_pretrained_info = f"select * from sessions where session_id='{session_id}';"
    cursor.execute(sql_pretrained_info)
    session_info = cursor.fetchall()

    if len(session_info) == 0:
        if weight_id is not None:
            sql_update_training = f"UPDATE ln_weight_info SET trainingYN='P' WHERE weightid={weight_id};"
            cursor.execute(sql_update_training)
            db.commit()
        return False
    return True


@app.route('/')
def kesco_training():
    return 'This is kesco training server'


@app.route('/train', methods=["POST", "GET"])
def train():
    global train_proc
    global db
    db_cursor = db.cursor()

    ## test
    weight_id = 3
    session_id = "0fihCiFwCOivDaKx6xbaBPbCUleQYhsH"


    # check session id
    if not check_session(db=db, session_id=session_id, weight_id=weight_id):
        return "Cannot verify session id"


    # training state
    sql_update_training = f"UPDATE ln_weight_info SET trainingYN='Y' WHERE weightid={weight_id};"
    db_cursor.execute(sql_update_training)
    db.commit()

    # get train arguments
    sql_pretrained_info = "select classificationfileurl, segmentationfileurl, createtm from ln_weight_info  where confirmYN ='Y';"
    db_cursor.execute(sql_pretrained_info)
    pretrained_info = db_cursor.fetchall()
    print(pretrained_info)

    # clf_path = pretrained_info[0][0].replace(".weights", ".pt")
    # mask_path = pretrained_info[0][1].replace(".weights", ".pt")
    clf_path = os.path.join(FILE_STORAGE_PATH, "weights", "efficientnet_b4_17_loss_0.70_acc_99.16.pt")
    mask_path = os.path.join(FILE_STORAGE_PATH, "weights", "mask_rcnn_27_AP_98.01.pt")
    data_path = FILE_STORAGE_PATH
    create_time = pretrained_info[0][2].strftime('%Y-%m-%d_%Hh%Mm%Ss')

    print("clf_model_path :", clf_path)
    print("mask_model_path :", mask_path)
    print("data_path :", data_path)
    print("create_time :", create_time)
    print("TRAIN_PATH :", TRAIN_PATH)


    # training process
    try:
        if isinstance(train_proc.poll(), int):
            train_proc = subprocess.Popen(['python', TRAIN_PATH, "--data_path", data_path,
                                           "--start_time", create_time, "--mask_path", mask_path,
                                           "--clf_path", clf_path], shell=True)
            train_proc.communicate()

        else:
            print("Training Process is running")
            return "Training Process is running"

    except:
        stop(session_id=session_id, weight_id=weight_id)
        raise ValueError("Training has stopped.")

    ## weight file check
    train_result = os.path.join(FILE_STORAGE_PATH, "train_results")

    # clf weight file path
    latest_clf = glob.glob(os.path.join(train_result, "efficientnet"))[-1]
    clf_weight_path = glob.glob(latest_clf+"\\*.weights")[-1]
    accuracy = clf_weight_path.split("_")[-1][:5]

    # mask weight file path
    latest_mask = glob.glob(os.path.join(train_result, "mask_rcnn"))[-1]
    mask_weight_path = glob.glob(latest_mask + "\\*.weights")[-1]

    # Database Update
    sql_update_training = f"UPDATE ln_weight_info SET classificationfileurl = {clf_weight_path}, " \
                          f"segmentationfileurl = {mask_weight_path}, accuracy = {str(accuracy)}, trainingYN='N' " \
                          f"WHERE weightid={weight_id};"
    db_cursor.execute(sql_update_training)
    db.commit()

    return "Training is Done"


@app.route('/stop', methods=["POST", "GET"])
def stop(session_id='0', weight_id=0):
    global train_proc
    global db

    ##
    weight_id = 3
    session_id = "0fihCiFwCOivDaKx6xbaBPbCUleQYhsH"
    ##

    if request.get_json(silent=True):
        json_data = request.get_json()
        session_id = json_data["sessionid"]
        weight_id = json_data["weightid"]


    # check session id
    if not check_session(db=db, session_id=session_id, weight_id=weight_id):
        return "Cannot verify session id"


    if not isinstance(train_proc.poll(), int):
        process = psutil.Process(train_proc.pid).children(recursive=True)
        for p in process:
            p.terminate()
        psutil.Process(train_proc.pid).terminate()

        gone, alive = psutil.wait_procs(process, timeout=10, callback=on_terminate)
        for p in alive:
            p.kill()

    db_cursor = db.cursor()
    sql_update_training = f"UPDATE ln_weight_info SET trainingYN = 'P' WHERE weightid={weight_id};"
    db_cursor.execute(sql_update_training)
    db.commit()

    return "Stopped"


@app.route('/progress', methods=["POST", "GET"])
def check_progress():
    global train_proc
    global db
    db_cursor = db.cursor()

    weight_id = 3
    session_id = "0fihCiFwCOivDaKx6xbaBPbCUleQYhsH"
    
    # get request
    if request.get_json(silent=True):
        json_data = request.get_json()
        session_id = json_data["sessionid"]
        weight_id = json_data["weightid"]
    
    # check session id
    if not check_session(db=db, session_id=session_id, weight_id=weight_id):
        return "Cannot verify session id"

    # get training info
    sql_training_info = "select classificationfileurl, segmentationfileurl from ln_weight_info where trainingYN='Y';"
    db_cursor.execute(sql_training_info)
    training_info = db_cursor.fetchall()

    print(training_info)

    clf_path = training_info[0][0]  ## .replace(".weights", ".pt")
    mask_path = training_info[0][1]  ## .replace(".weights", ".pt")
    # mask_folder = "E:\\work\\kesco\\file_storage\\train_results\\mask_rcnn\\2022-01-20_10h10m58s\\best_model_mAP_0.74.weights"
    # clf_folder = "E:\\work\\kesco\\file_storage\\train_results\\efficientnet\\2022-01-20_10h10m58s\\best_model_acc_99.41.weights"

    mask_log = os.path.join(os.path.dirname(mask_folder), "metrics.json")
    clf_log = os.path.join(os.path.dirname(clf_folder), "train.log")

    # default percentage
    percentage = 0

    if os.path.isfile(clf_log):
        logs = load_txt_arr(clf_log)
        epochs = [x.split(' ')[2] for x in logs if "Train Epoch" in x]
        latest_epoch = int(epochs[-1])
        percentage = int(latest_epoch / 200 * 100 * 0.5) + 50  # total epochs = 200, +50% : after mask rcnn complete
        return percentage

    elif os.path.isfile(mask_log):
        metrics = load_json_arr(mask_log)
        iters = [x['iteration'] for x in metrics if 'total_loss' in x]
        lastest_iter = iters[-1]
        percentage = int(lastest_iter / 20000 * 100 * 0.5)  # total iteration = 20000
        return percentage

    return percentage



@app.route('/autolabel', methods=["POST", "GET"])
def autolabelling():
    global db
    db_cursor = db.cursor()


    ###
    session_id = "0fihCiFwCOivDaKx6xbaBPbCUleQYhsH"
    weight_id = 3

    specimen_id = '한-전북-202201001'
    pseq = 1
    seq = 1
    weight_id = 3
    ###

    # get request
    if request.get_json(silent=True):
        json_data = request.get_json()

        session_ids = json_data["sessionid"]
        weight_ids = json_data["weightid"]
        specimen_ids = json_data["specimenid"]
        pseqs = json_data["pseq"]
        seqs = json_data["seq"]

    
    if not check_session(db=db, session_id=session_id):
        return "Cannot verify session id"

    # get labelling list
    sql_labelling_list = f"select maskpath, maskrip from ln_shot_history " \
                         f"where specimenid='{specimen_id}' and pseq={pseq} and seq={seq} and weightid='{weight_id}';"
    db_cursor.execute(sql_labelling_list)
    labelling_list = db_cursor.fetchall()

    binary_mask_path = labelling_list[0][0]
    RIP_path = labelling_list[0][1]

    json_data = autolabel(RIP_path=RIP_path, binary_mask_path=binary_mask_path)

    # save json
    json_path = "append.json"
    with open(json_path, "w") as json_file:
        json.dump(json_data, json_file)

    # update database
    sql_labelling_list = f"UPDATE ln_shot_history SET jsonpath = {json_path} WHERE " \
                         f"specimenid='{specimen_id}' and pseq={pseq} and seq={seq} and weightid='{weight_id}';"
    db_cursor.execute(sql_labelling_list)
    db.commit()

    return json_data

    
if __name__=="__main__":
    app.run(host='0.0.0.0', port=80)
