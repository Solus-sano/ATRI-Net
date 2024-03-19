import json, time, os, logging
import rarfile
import zipfile
from common import Utils, TideoHelper
from common.ArtistServer import Artist

def find_root(savepath, slidelist):
    embryolist = []
    for slide in slidelist:
        for root, dirs, files in os.walk(savepath):
            for dirname in dirs:
                if '_WELL' in dirname and slide in dirname:
                    embryolist.append(os.path.join(root, dirname))
    return embryolist

if __name__ == '__main__':
    app = Artist()
    while True:
        try:
            data = TideoHelper.query('ARTRCT', 'SubmitJob')
            if data is not None and '_id' in data:
                logging.info("receive AI job:"+data)
                data = json.loads(data)
                basepath = os.path.join('/root/autodl-tmp/static', data["pid"], data["_id"])
                os.makedirs(basepath, exist_ok=True)
                if 'timelapse' in data:
                    filename = data["timelapse"].split('/').pop()
                    
                    suffix = filename.split('.').pop().lower()
                    savefile = os.path.join(basepath, filename)
                    savepath = os.path.splitext(savefile)[0]
                    Utils.download(data["timelapse"], savefile)
                    
                    try:
                        with zipfile.ZipFile(savefile, mode='r') as fp:
                            fp.extractall(savepath)
                    except Exception as e:
                        logging.warning("以zip格式解压文件失败")
                        try:
                            rar = rarfile.RarFile(savefile, mode='r')
                            rar.extractall(savepath)
                            rar.close()
                        except Exception as e:
                            logging.warning("以rar格式解压文件失败")
                            raise Exception('文件解压失败')
                    try:
                        slidelist = sorted(data["slide"].split('\n'))
                        print(slidelist)
                        res = []
                        embryolist = sorted(find_root(savepath, slidelist))
                        print(embryolist)
                        for embryo in embryolist:
                            name = os.path.basename(embryo)
                            tmp = name.split('_')
                            eid = tmp[-1]
                            mid = tmp[1] + '_' + tmp[2] + '_D'
                            for root, dirs, files in os.walk(embryo):
                                if 'F0' in dirs:
                                    probility = app.calculate(femaleAge=data["age"], maleAge=data["hage"],
                                                              buYun=data["acyesis"], yiWei=data["translocation"],
                                                              img_path=root, eid=eid, mid=mid)
                                    reply = {"eid": eid, "mid": mid, "pid": data["_id"], "status": 200, "probility": probility, "ai":0, "error": ""}
                                    break
                            res.append(reply)

                        sorted_res = sorted(res, key=lambda a:a['probility'],reverse=True)
                        print(sorted_res)
                        for idx, i in enumerate(sorted_res):
                            for j in res:
                                if j["eid"]==i["eid"] and j["mid"]==i["mid"]:
                                    j["ai"]=idx+1
                                    break
                        resp = TideoHelper.updatPatient(data["_id"], res)
                        print(resp)
                        print('update embryo result')
                    except Exception as e:
                        print(e)
                        res = []

                print('='*25, '  End  ', '='*25)
        except Exception as e:
            print(e)
        time.sleep(1)