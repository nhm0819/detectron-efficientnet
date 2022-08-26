import os
import subprocess
import glob
import json
import mysql.connector
from flask import Flask, request, g, abort
from flask_cors import CORS
from main_utils import check_session, load_txt_arr, load_json_arr, process_kill
from labelling import Labeller
import datetime
import shutil
import logging
import logging.handlers

# flask 실행
app = Flask(__name__)
CORS(app)

# flask의 모든 request 전에 거쳐 갈 함수
@app.before_request
def before_request():
    print(request.remote_addr)
    
    # logging handler
    log_level = logging.DEBUG

    # 실행중인 handler 종료
    for handler in app.logger.handlers:
        app.logger.removeHandler(handler)

    # logs 폴더 생성
    root = os.path.dirname(os.path.abspath(__file__))
    logdir = os.path.join(root, 'logs')
    if not os.path.exists(logdir):
        os.mkdir(logdir)
    log_file = os.path.join(logdir, 'app.log')

    # stream handler 선언
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(log_level)

    # file handler 선언
    fileMaxByte = 1024 * 1024 * 100
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(log_level)

    # handler format 선언
    defaultFormatter = logging.Formatter('[%(asctime)s] %(levelname)s in %(module)s: %(message)s')
    file_handler.setFormatter(defaultFormatter)
    stream_handler.setFormatter(defaultFormatter)

    # flask에 handler 추가
    app.logger.addHandler(stream_handler)
    app.logger.addHandler(file_handler)
    app.logger.setLevel(log_level)

    # DB 서버, Inference 서버, 내 서버 아니면 abort
    if request.remote_addr not in [os.getenv("KESCO-DB-HOST"), os.getenv("KESCO-INFERENCE"), '127.0.0.1']:
        abort(403)  # Forbidden


    # 환경변수에서 file storage 주소 획득
    g.FILE_STORAGE_PATH = os.getenv("KESCO-DATASET-PATH")
    g.TRAIN_RESULTS_PATH = os.path.join(g.FILE_STORAGE_PATH, "train_results")

    # db connect
    g.db = mysql.connector.connect(host=os.getenv("KESCO-DB-HOST"), port=os.getenv("KESCO-DB-PORT"), database=os.getenv("KESCO-DB-NAME"), user=os.getenv("KESCO-DB-USER"),
                               password=os.getenv("KESCO-DB-PASSWORD"))

    # epochs 설정
    g.epochs = '200'
    g.mask_iters = '20000'


# app 종료 시 db 종료
@app.teardown_appcontext
def teardown_appcontext(response):
    g.db.close()


# train 함수
@app.route('/train', methods=["POST", "GET"])
def train():

    # db cursor 실행
    db = g.db
    cursor = db.cursor()

    # request file 체크
    if request.get_json(silent=True):
        json_data = request.get_json()
        session_id = json_data["sessionid"]
        weight_id = json_data["weightid"]
    else:
        return "Cannot read request"

    # check session id
    if not check_session(db=db, session_id=session_id):
        return "Cannot verify session id"

    # check training state
    sql_training_list = f"select weightid, trainingYN, pid from ln_weight_info where trainingYN='Y';"
    cursor.execute(sql_training_list)
    training_list = cursor.fetchall()
    if len(training_list) > 0:
        return "Training process is running"


    # get train arguments (현재 사용하는 weight file에 대한 정보)
    sql_pretrained_info = "select classificationfileurl, segmentationfileurl from ln_weight_info where confirmYN ='Y';"
    cursor.execute(sql_pretrained_info)
    pretrained_info = cursor.fetchall()
    print(pretrained_info)

    # default model weights
    clf_orig = os.path.join(g.FILE_STORAGE_PATH, "weights", "efficientnet_over_current_acc_98.55.pt")
    mask_orig = os.path.join(g.FILE_STORAGE_PATH, "weights", "mask_rcnn_26_AP_98.33.pt")

    # 현재 사용하는 weight 불러오기
    try:
        clf_path = pretrained_info[0][0].replace(".weights", ".pt").replace("/", "\\")
        mask_path = pretrained_info[0][1].replace(".weights", ".pt").replace("/", "\\")
    except:
        clf_path = clf_orig
        mask_path = mask_orig

    # weight 파일이 없을 시 가장 첫 weight 파일 입력
    if not os.path.isfile(clf_path):
        clf_path = clf_orig
    if not os.path.isfile(mask_path):
        mask_path = mask_orig

    # db 조회
    sql_weight_info = f"select createtm, datapath from ln_weight_info where weightid={weight_id};"
    cursor.execute(sql_weight_info)
    weight_info = cursor.fetchall()

    # dataset 생성 시간 및 path 확인
    create_time = weight_info[0][0].strftime('%Y-%m-%d_%Hh%Mm%Ss')
    data_path = weight_info[0][1]

    # 학습 코드 경로 저장
    train_file = os.path.join(os.getcwd(), "auto_train.py")

    print("weight_id :", weight_id)
    print("clf_model_path :", clf_path)
    print("mask_model_path :", mask_path)
    print("data_path :", data_path)
    print("create_time :", create_time)
    print("TRAIN_PATH :", train_file)

    # start training process
    train_proc = subprocess.Popen(['python', train_file, "--file_storage_path", g.FILE_STORAGE_PATH, "--data_path", data_path,
                                    "--start_time", create_time, "--mask_path", mask_path, "--clf_path", "", #clf_path,
                                   "--epochs", g.epochs, "--mask_iters", g.mask_iters
                                   ], shell=True)

    # hyper parameter 조절 예시
    # train_proc = subprocess.Popen(['python', train_file, "--file_storage_path", g.FILE_STORAGE_PATH, "--data_path", data_path,
    #                             "--start_time", create_time, "--mask_path", mask_path, "--clf_path", clf_path,
    #                             "--mask_iters", "40", "--mask_checkpoint", "20", "--epochs", "5",
    #                             "--val_per_epochs", "3", "--test_per_epochs", "3", "--testing"], shell=True)

    train_pid = train_proc.pid

    # training 상태 업데이트
    sql_update_training = f"UPDATE ln_weight_info SET trainingYN='Y', pid={train_pid} WHERE weightid={weight_id};"
    cursor.execute(sql_update_training)
    db.commit()

    # db close
    cursor.close()
    db.close()

    # train process가 끝날 때 까지 대기
    train_proc.communicate()

    # weight 파일 확인 후 없을 시 데이터 삭제 후 학습 실패
    check_training_weights = os.path.join(g.TRAIN_RESULTS_PATH, "efficientnet", create_time, "*.weights")
    if len(glob.glob(check_training_weights)) == 0:
        training_stop(session_id=session_id, weight_id=weight_id, training_error=True)
        shutil.rmtree(os.path.join(g.TRAIN_RESULTS_PATH, "efficientnet", create_time))
        shutil.rmtree(os.path.join(g.TRAIN_RESULTS_PATH, "mask_rcnn", create_time))
        
        # db close
        if db.is_connected():
            cursor.close()
            db.close()

        # raise ValueError("Training has stopped.")
        
        return "Training Error occured"

    # 학습 종료
    training_stop(session_id=session_id, terminate_mode=True)

    # clf weight file path
    clf_folder = os.path.join(g.TRAIN_RESULTS_PATH, "efficientnet", create_time)
    clf_weight_path = glob.glob(os.path.join(clf_folder, "*.weights"))[-1].replace("\\", "/")
    accuracy = clf_weight_path.split("_")[-1][:-8]

    # mask weight file path
    mask_folder = os.path.join(g.TRAIN_RESULTS_PATH, "mask_rcnn", create_time)
    mask_weight_path = glob.glob(os.path.join(mask_folder, "*.weights"))[-1].replace("\\", "/")

    # db connection
    g.db = mysql.connector.connect(host=os.getenv("KESCO-DB-HOST"), port=os.getenv("KESCO-DB-PORT"), database=os.getenv("KESCO-DB-NAME"), user=os.getenv("KESCO-DB-USER"),
                               password=os.getenv("KESCO-DB-PASSWORD"))
    db = g.db
    cursor = db.cursor()

    # Database Update
    sql_update_training = f"UPDATE ln_weight_info SET classificationfileurl = '{clf_weight_path}', " \
                          f"segmentationfileurl = '{mask_weight_path}', accuracy = '{accuracy}', trainingYN='N' " \
                          f"WHERE weightid={weight_id};"
    cursor.execute(sql_update_training)
    db.commit()

    cursor.close()

    return "Training is Done"


# 학습 중단 기능
@app.route('/stop', methods=["POST", "GET"])
def training_stop(session_id='0', weight_id=-1, training_error=False, terminate_mode=False):

    db = g.db
    if not db.is_connected():
        g.db = mysql.connector.connect(host=os.getenv("KESCO-DB-HOST"), port=os.getenv("KESCO-DB-PORT"), database=os.getenv("KESCO-DB-NAME"), user=os.getenv("KESCO-DB-USER"),
                               password=os.getenv("KESCO-DB-PASSWORD"))
        db = g.db
    
    cursor = db.cursor()

    if request.get_json(silent=True):
        json_data = request.get_json()
        session_id = json_data["sessionid"]
        # weight_id = json_data["weightid"]
    else:
        return "Cannot read request"

    if not check_session(db=db, session_id=session_id):
        return "Cannot verify session id"

    # terminate train processes
    if training_error: # 오류 발생 시 학습 중단
        # Kill the specific training process (when training error occured)
        sql_training_proc = f"select createtm, pid from ln_weight_info where weightid={weight_id};"
        cursor.execute(sql_training_proc)
        training_list = cursor.fetchall()
        create_time = training_list[0][0].strftime('%Y-%m-%d_%Hh%Mm%Ss')
        train_pid = training_list[0][1]

        try:
            process_kill(train_pid)
        except:
            print(f"process ID has terminated.(pid={train_pid})")
            pass
            
        # 학습 기록 제거
        clf_folder = os.path.join(g.TRAIN_RESULTS_PATH, "efficientnet", create_time)
        mask_folder = os.path.join(g.TRAIN_RESULTS_PATH, "mask_rcnn", create_time)
        try:
            shutil.rmtree(clf_folder)
            shutil.rmtree(mask_folder)
        except:
            pass

        # db update
        sql_update_training = f"UPDATE ln_weight_info SET trainingYN='P' WHERE weightid={weight_id};"
        cursor.execute(sql_update_training)
        db.commit()

    else: # 선택적 학습 중단
        # Kill all training processes
        sql_training_proc = f"select createtm, pid from ln_weight_info where trainingYN='Y';"
        cursor.execute(sql_training_proc)
        training_list = cursor.fetchall()

        for training in training_list:
            create_time = training[0].strftime('%Y-%m-%d_%Hh%Mm%Ss')
            train_pid = training[1]
            
            try:
                process_kill(train_pid)
            except:
                print(f"process ID has terminated.(pid={train_pid})")
                pass
        if terminate_mode: # 학습 종료 후 db 수정
            sql_update_training = f"UPDATE ln_weight_info SET trainingYN='N' WHERE trainingYN='Y';"
            cursor.execute(sql_update_training)
            db.commit()
        else: # 학습 중단 버튼
            sql_update_training = f"UPDATE ln_weight_info SET trainingYN='P' WHERE trainingYN='Y';"
            cursor.execute(sql_update_training)
            db.commit()

    cursor.close()

    return "Stopped"


# 학습 진행률
@app.route('/progress', methods=["POST", "GET"])
def check_progress():
    db = g.db
    cursor = db.cursor()

    if request.get_json(silent=True):
        json_data = request.get_json()
        session_id = json_data["sessionid"]
        # weight_id = json_data["weightid"]
    else:
        return "Cannot read request"

    if not check_session(db=db, session_id=session_id):
        return "Cannot verify session id"

    # default percentage
    percentage = '0'


    # get training info
    sql_training_info = "select createtm from ln_weight_info where trainingYN='Y';"
    cursor.execute(sql_training_info)
    createtm_info = cursor.fetchall()
    if len(createtm_info) == 0: # 정보 없을 시
        return percentage

    create_time = createtm_info[0][0].strftime('%Y-%m-%d_%Hh%Mm%Ss')

    # clf weight file path
    clf_folder = os.path.join(g.TRAIN_RESULTS_PATH, "efficientnet", create_time)
    mask_folder = os.path.join(g.TRAIN_RESULTS_PATH, "mask_rcnn", create_time)

    # log file path
    clf_log = os.path.join(clf_folder, "train.log")
    mask_log = os.path.join(mask_folder, "metrics.json")

    # Find progress of training using training logs.
    if os.path.isdir(mask_folder):
        percentage = '1' # training is started
        if os.path.isfile(mask_log): # mask rcnn 확인 후 efficientnet 확인
            if os.path.isfile(clf_log):
                logs = load_txt_arr(clf_log)
                epochs = [x.split(' ')[2] for x in logs if "Train Epoch" in x]
                latest_epoch = 0
                if len(epochs) != 0:
                    latest_epoch = int(epochs[-1]) # 가장 최근 epoch 확인
                percentage = str(int(latest_epoch / 200 * 100 * 0.5) + 50)  # total epochs = 200, +50% : after mask rcnn complete
                # percentage = str(int(latest_epoch / int(g.epochs) * 100 * 0.5) + 50)
                return percentage

            metrics = load_json_arr(mask_log)
            iters = [x['iteration'] for x in metrics if 'total_loss' in x]
            latest_iter = 0
            if len(iters) != 0:
                latest_iter = iters[-1] # 가장 최근 iters 확인
            percentage = str(int(latest_iter / 20000 * 100 * 0.5) + 1)  # total iteration = 20000
            # percentage = str(int(latest_iter / int(g.mask_iters) * 100 * 0.5) + 1)
            return percentage

        return percentage

    cursor.close()

    return percentage


# autolabelling 기능
@app.route('/autolabel', methods=["POST", "GET"])
def autolabel():

    db = g.db
    cursor = db.cursor()

    if request.get_json(silent=True):
        json_data = request.get_json()
        session_id = json_data["sessionid"]
        weight_id = json_data["weightid"]
    else:
        return "Cannot read request"

    if not check_session(db=db, session_id=session_id):
        return "Cannot verify session id"

    # labelling.py 의 Labeller class
    auto_labeller = Labeller(db, weight_id, nms_cnt=3, num_classes=26, wire_class_num=25,
                 image_shape=(720,1280), epsilon_rate=0.003)

    try:
        auto_labeller.labelling() # labelling 실행
    except:
        return "Error in autolabelling"

    print("Autolabelling has done")
    return "Autolabelling has done"



if __name__=="__main__":
    app.run(host='0.0.0.0', port=80)
    # serve(app, host='0.0.0.0', port=80, threads=4)
