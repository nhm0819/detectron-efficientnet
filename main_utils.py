import psutil
import json

def check_session(db, session_id):
    cursor = db.cursor()
    sql_pretrained_info = f"select * from sessions where session_id='{session_id}';"
    cursor.execute(sql_pretrained_info)
    session_info = cursor.fetchall()

    cursor.close()

    if len(session_info) == 0:
        # if weight_id is not None:
        #     sql_update_training = f"UPDATE ln_weight_info SET trainingYN='P' WHERE weightid={weight_id};"
        #     cursor.execute(sql_update_training)
        #     db.commit()
        return False
    return True


def load_txt_arr(txt_path):
    f = open(txt_path, 'r')
    lines = f.readlines()
    f.close()
    return lines

def load_json_arr(json_path):
    lines = []
    with open(json_path, 'r') as f:
        for line in f:
            lines.append(json.loads(line))
    return lines

def on_terminate(proc):
    print("process {} terminated with exit code {}".format(proc, proc.returncode))

def process_kill(pid):
    process = psutil.Process(pid).children(recursive=True)
    for p in process:
        p.terminate()
    psutil.Process(pid).terminate()

    gone, alive = psutil.wait_procs(process, timeout=10, callback=on_terminate)
    for p in alive:
        p.kill()