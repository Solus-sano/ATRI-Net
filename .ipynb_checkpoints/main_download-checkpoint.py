import json, time, os
import rarfile
import zipfile
from common import Utils, TideoHelper


def find_root(savepath):
    embryolist = []
    for root, dirs, files in os.walk(savepath):
        for dir in dirs:
            if '_WELL' in dir:
                embryolist.append(os.path.join(root,dir))
    return embryolist

if __name__ == '__main__':
    while True:
        try:
            data = TideoHelper.query('ARTRCT', 'DownloadJob')
            if data is not None and '_id' in data:
                print('='*25, '  Download  ', '='*25)
                print('【{}】receive download job'.format(Utils.get_now_time()))
                print(data)
                data = json.loads(data)
                basepath = os.path.join('/root/autodl-tmp/static', data["pid"], data["_id"])
                os.makedirs(basepath, exist_ok=True)
                if 'timelapse' in data:
                    filename = data["timelapse"]["url"].split('/').pop()
                    suffix = filename.split('.').pop().lower()
                    savefile = os.path.join(basepath, filename)
                    savepath = os.path.splitext(savefile)[0]

                    Utils.download(data["timelapse"]["url"], savefile)
                    if suffix == 'rar':
                        rar = rarfile.RarFile(savefile, mode='r')
                        rar.extractall(savepath)
                        rar.close()
                    elif suffix == 'zip':
                        with zipfile.ZipFile(savefile, mode='r') as fp:
                            fp.extractall(savepath)
                    else:
                        reply = {'id': data["_id"], 'data': { 'status': 400, 'error': '请把照片压缩成rar或zip格式'}}
                    try:
                        res = []
                        embryolist = sorted(find_root(savepath))
                        for embryo in embryolist:
                            name = os.path.basename(embryo)
                            tmp = name.split('_')
                            eid = tmp[-1]
                            mid = tmp[1]+'_'+tmp[2]+'_D'
                            for root, dirs, files in os.walk(embryo):
                                if 'F0' in dirs:
                                    reply = {"eid": eid, "mid": mid, "pid": data["_id"]}
                                    break
                            res.append(reply)
                        print(res)
                        resp = TideoHelper.updatPatient(data["_id"], res)
                        print(resp)
                        print('【{}】upload embryo list over'.format(Utils.get_now_time()))
                    except Exception as e:
                        print(e)
                        res = []

                print('='*25, '  End  ', '='*25)
        except Exception as e:
            print(e)
            print('cannot connect the server')
        time.sleep(3)