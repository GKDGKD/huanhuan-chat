import os
from pprint import pprint
from litie.pipelines import UIEPipeline
from litie.pipelines import RelationExtractionPipeline

os.environ['HTTPS_PROXY']    = 'http://127.0.0.1:7890'
os.environ["HTTP_PROXY"]     = 'http://127.0.0.1:7890'


# # 关系抽取
# schema = {'竞赛名称': ['主办方', '承办方', '已举办次数']}
# # uie-base模型已上传至huggingface，可自动下载，其他模型只需提供模型名称将自动进行转换
# uie = UIEPipeline("xusenlin/uie-base", schema=schema)
# pprint(uie("2022语言与智能技术竞赛由中国中文信息学会和中国计算机学会联合主办，百度公司、中国中文信息学会评测工作委员会和中国计算机学会自然语言处理专委会承办，已连续举办4届，成为全球最热门的中文NLP赛事之一。")) # Better print results using pprint

# # 实体识别
# schema = ['时间', '选手', '赛事名称'] 
# uie = UIEPipeline("xusenlin/uie-base", schema=schema)
# pprint(uie("2月8日上午北京冬奥会自由式滑雪女子大跳台决赛中中国选手谷爱凌以188.25分获得金牌！")) # Better print results using pprint

## 关系抽取
text = """
甄嬛（钮祜禄•甄嬛），小说《后宫·甄嬛传》和电视剧《甄嬛传》中的女一号，核心女主角。
原名甄玉嬛，嫌玉字俗气而改名甄嬛，为汉人甄远道之女，后被雍正赐姓钮祜禄氏，抬旗为满洲上三旗，获名“钮祜禄·甄嬛”。
同沈眉庄、安陵容参加选秀，因容貌酷似纯元皇后而被选中。入宫后面对华妃的步步紧逼，沈眉庄被冤、安陵容变心，从偏安一隅的青涩少女变成了能引起血雨腥风的宫斗老手。雍正发现年氏一族的野心后令其父甄远道剪除，甄嬛也于后宫中用她的连环巧计帮皇帝解决政敌，故而深得雍正爱待。几经周折，终于斗垮了嚣张跋扈的华妃。
甄嬛封妃时遭皇后宜修暗算，被皇上嫌弃，生下女儿胧月后心灰意冷，自请出宫为尼。然得果郡王爱慕，二人相爱，得知果郡王死讯后立刻设计与雍正再遇，风光回宫。此后甄父冤案平反、甄氏复起，她也生下双生子，在滴血验亲等各种阴谋中躲过宜修的暗害，最后以牺牲自己亲生胎儿的方式扳倒了幕后黑手的皇后。
但雍正又逼甄嬛毒杀允礼，以测试甄嬛真心，并让已经生产过孩子的甄嬛去准格尔和亲。甄嬛遂视皇帝为最该毁灭的对象，大结局道尽“人类的一切争斗，皆因统治者的不公不义而起”，并毒杀雍正。四阿哥弘历登基为乾隆，甄嬛被尊为圣母皇太后，权倾朝野，在如懿传中安度晚年。
"""

# 实体识别
schema     = ['小说', '人物', '外号']
uie_entity = UIEPipeline("xusenlin/uie-base", schema=schema, device='gpu')
res_entity = uie_entity(text)
pprint(res_entity) # Better print results using pprint

# 关系抽取
schema = {'人物关系': ['父亲','爱人', '朋友', '敌人', '女儿']}
# uie-base模型已上传至huggingface，可自动下载，其他模型只需提供模型名称将自动进行转换
uie          = UIEPipeline("xusenlin/uie-base", schema=schema, device='gpu')
res_relation = uie(text)
pprint(res_relation) # Better print results using pprint

# 筛选关系，只保留可能性大于设定阈值的关系
relation_threshold = 0.6
r = []
for item in res_relation[0]['人物关系']:
    if item['probability'] > relation_threshold:
        r.append(item)
pprint(r)
