# encoding: utf-8
from agent import MahjongGBAgent
from collections import defaultdict
import numpy as np

try:
    from MahjongGB import MahjongFanCalculator
except:
    print('MahjongGB library required! Please visit https://github.com/ailab-pku/PyMahjongGB for more information.')
    raise

class FeatureAgent(MahjongGBAgent):
    
    '''
    observation: 6*4*9#修改以支持更加完整的训练
        (men+quan+hand4)*4*9
    action_mask: 235
        pass1+hu1+discard34+chi63(3*7*3)+peng34+gang34+angang34+bugang34
    '''
    
    OBS_SIZE = 135
    ACT_SIZE = 235
    #表示各种特征的起始位置
    #圈风1 门风1 自己手牌4 四个人的副露（吃、碰、明杠）12 自己暗杠1 剩余牌4 弃牌历史28*4((34*4-13*4)/4=22) 
    OFFSET_OBS = {
        'SEAT_WIND' : 0,
        'PREVALENT_WIND' : 1,
        'HAND' : 2,
        'PACKS' : 6,
        'SELF_ANGANG' : 18,
        'REMAIN' : 19,
        'PLAYED' : 23
    }
    OFFSET_ACT = {
        'Pass' : 0,
        'Hu' : 1,
        'Play' : 2,#打牌是2维
        'Chi' : 36,#吃是63维
        'Peng' : 99,#碰是34维
        'Gang' : 133,#杠是34维
        'AnGang' : 167,
        'BuGang' : 201
    }
    TILE_LIST = [
        *('W%d'%(i+1) for i in range(9)),#万
        *('T%d'%(i+1) for i in range(9)),#条
        *('B%d'%(i+1) for i in range(9)),#饼
        *('F%d'%(i+1) for i in range(4)),#风
        *('J%d'%(i+1) for i in range(3))#中发白
    ]
    OFFSET_TILE = {c : i for i, c in enumerate(TILE_LIST)}#表示每张牌的编号
    
    def __init__(self, seatWind):
        self.seatWind = seatWind # 表示自己的门风，即自己的座次
        self.packs = [[] for i in range(4)] # 手牌
        self.history = [[] for i in range(4)]
        self.tileWall = [21] * 4
        self.shownTiles = defaultdict(int)#
        self.wallLast = False
        self.isAboutKong = False#这是什么？
        self.obs = np.zeros((self.OBS_SIZE, 36))
        self.obs[self.OFFSET_OBS['REMAIN'] : self.OFFSET_OBS['REMAIN'] + 4] = np.ones(36)
        self.obs[self.OFFSET_OBS['SEAT_WIND'], self.OFFSET_TILE['F%d' % (self.seatWind + 1)]] = 1#用牌标记好圈风
    
    '''
    Wind 0..3
    Deal XX XX ...
    Player N Draw
    Player N Gang
    Player N(me) AnGang XX
    Player N(me) Play XX
    Player N(me) BuGang XX
    Player N(not me) Peng
    Player N(not me) Chi XX
    Player N(not me) AnGang
    
    Player N Hu
    Huang
    Player N Invalid
    Draw XX
    Player N(not me) Play XX
    Player N(not me) BuGang XX
    Player N(me) Peng
    Player N(me) Chi XX
    '''
    def request2obs(self, request):
        t = request.split()
        #下面几个选择句可以处理玩家自己的信息
        if t[0] == 'Wind':
            self.prevalentWind = int(t[1])
            self.obs[self.OFFSET_OBS['PREVALENT_WIND'], self.OFFSET_TILE['F%d' % (self.prevalentWind + 1)]] = 1
            #标记门风
            return
        if t[0] == 'Deal':
            self.hand = t[1:]
            self._hand_embedding_update()

            for tile in t[1:]:
                self.obs[self.OFFSET_OBS['REMAIN'], self.OFFSET_TILE[tile]] = 0

            return
        if t[0] == 'Huang':#这是什么？：荒庄：摸完牌也没胡
            self.valid = []
            return self._obs()
        if t[0] == 'Draw':
            # Available: Hu, Play, AnGang, BuGang
            self.tileWall[0] -= 1#始终是0吗？
            self.wallLast = self.tileWall[1] == 0#？
            tile = t[1]
            self.valid = []#表示可行动作
            if self._check_mahjong(tile, isSelfDrawn = True, isAboutKong = self.isAboutKong):
                self.valid.append(self.OFFSET_ACT['Hu'])
            self.isAboutKong = False
            self.hand.append(tile)
            self._hand_embedding_update()
            for tile in set(self.hand):
                #可行动作中加入可以打的牌
                self.valid.append(self.OFFSET_ACT['Play'] + self.OFFSET_TILE[tile])
                #加入暗杠
                if self.hand.count(tile) == 4 and not self.wallLast and self.tileWall[0] > 0:
                    self.valid.append(self.OFFSET_ACT['AnGang'] + self.OFFSET_TILE[tile])
            if not self.wallLast and self.tileWall[0] > 0:
                for packType, tile, offer in self.packs[0]:#对手里所有的明牌逐一检查，加上补杠
                    if packType == 'PENG' and tile in self.hand:
                        self.valid.append(self.OFFSET_ACT['BuGang'] + self.OFFSET_TILE[tile])
            return self._obs() # 包括当前的状态以及可以实现的动作掩码
        # Player N Invalid/Hu/Draw/Play/Chi/Peng/Gang/AnGang/BuGang XX
        # 下面几个条件句是处理别的玩家的信息

        p = (int(t[1]) + 4 - self.seatWind) % 4 # 将全局的座次转换为当前视角下的座次，0-自己，1-下家，2-对家，3-下家
        if t[2] == 'Draw':
            self.tileWall[p] -= 1
            self.wallLast = self.tileWall[(p + 1) % 4] == 0 # 下家的牌？
            return
        if t[2] == 'Invalid':
            self.valid = []
            return self._obs()
        if t[2] == 'Hu':
            self.valid = []
            return self._obs()
        if t[2] == 'Play':
            self.tileFrom = p  # 记录当前的牌是谁打出的
            self.curTile = t[3]  # 记录打出的牌是哪一张(current_Tile)
            self.shownTiles[self.curTile] += 1 #表示局面上出现过的所有牌的数目
            self._history_emdedding_update()
            self.history[p].append(self.curTile)
            if p == 0:
                self.hand.remove(self.curTile)
                self._hand_embedding_update()
                return
            else:
                # Available: Hu/Gang/Peng/Chi/Pass
                self.valid = []
                if self._check_mahjong(self.curTile):
                    self.valid.append(self.OFFSET_ACT['Hu'])
                if not self.wallLast:
                    if self.hand.count(self.curTile) >= 2:
                        self.valid.append(self.OFFSET_ACT['Peng'] + self.OFFSET_TILE[self.curTile])
                        if self.hand.count(self.curTile) == 3 and self.tileWall[0]:
                            self.valid.append(self.OFFSET_ACT['Gang'] + self.OFFSET_TILE[self.curTile])
                    color = self.curTile[0]
                    #可以吃
                    if p == 3 and color in 'WTB':
                        num = int(self.curTile[1])
                        tmp = []
                        for i in range(-2, 3): tmp.append(color + str(num + i))
                        if tmp[0] in self.hand and tmp[1] in self.hand:
                            self.valid.append(self.OFFSET_ACT['Chi'] + 'WTB'.index(color) * 21 + (num - 3) * 3 + 2)#吃最后一张，最后一个数字需要同时给出中间牌和偏移的信息
                            # 以两万三万四万为例，num-x表示都平移到1，乘3表示都乘到三万的前一位，加上不同的值分别表示第一张、第二张和第三张
                        if tmp[1] in self.hand and tmp[3] in self.hand:
                            self.valid.append(self.OFFSET_ACT['Chi'] + 'WTB'.index(color) * 21 + (num - 2) * 3 + 1)#吃中间那一张
                        if tmp[3] in self.hand and tmp[4] in self.hand:
                            self.valid.append(self.OFFSET_ACT['Chi'] + 'WTB'.index(color) * 21 + (num - 1) * 3)#吃第一张
                self.valid.append(self.OFFSET_ACT['Pass'])
                return self._obs()
        if t[2] == 'Chi':
            tile = t[3]
            color = tile[0]
            num = int(tile[1])
            self.packs[p].append(('CHI', tile, int(self.curTile[1]) - num + 2))#给信息的时候只会显示顺子中间的那张牌

            self.obs[self.OFFSET_OBS['PACKS'] + p * 3, self.OFFSET_TILE[tile]] = 1

            self.shownTiles[self.curTile] -= 1
            for i in range(-1, 2):
                self.shownTiles[color + str(num + i)] += 1#局面上的明牌
            self._history_emdedding_update()
            self.wallLast = self.tileWall[(p + 1) % 4] == 0
            if p == 0:
                # Available: Play
                self.valid = []
                self.hand.append(self.curTile)
                for i in range(-1, 2):
                    self.hand.remove(color + str(num + i))#暗牌
                self._hand_embedding_update()
                for tile in set(self.hand):
                    self.valid.append(self.OFFSET_ACT['Play'] + self.OFFSET_TILE[tile])
                return self._obs()
            else:
                return
        if t[2] == 'UnChi':###??????
            tile = t[3]
            color = tile[0]
            num = int(tile[1])
            self.packs[p].pop()
            self.obs[self.OFFSET_OBS['PACKS'] + p * 3, self.OFFSET_TILE[tile]] = 0
            self.shownTiles[self.curTile] += 1
            for i in range(-1, 2):
                self.shownTiles[color + str(num + i)] -= 1
            self._history_emdedding_update()
            if p == 0:
                for i in range(-1, 2):
                    self.hand.append(color + str(num + i))
                self.hand.remove(self.curTile)
                self._hand_embedding_update()
            return
        if t[2] == 'Peng':
            self.packs[p].append(('PENG', self.curTile, (4 + p - self.tileFrom) % 4))

            self.obs[self.OFFSET_OBS['PACKS'] + p * 3 + 1, self.OFFSET_TILE[self.curTile]] = 1    

            self.shownTiles[self.curTile] += 2 #新亮出了两张牌
            self._history_emdedding_update()
            self.wallLast = self.tileWall[(p + 1) % 4] == 0
            if p == 0:
                # Available: Play
                self.valid = []
                for i in range(2):
                    self.hand.remove(self.curTile)
                self._hand_embedding_update()
                for tile in set(self.hand):
                    self.valid.append(self.OFFSET_ACT['Play'] + self.OFFSET_TILE[tile])
                return self._obs()
            else:
                return
        if t[2] == 'UnPeng':
            self.packs[p].pop()

            self.obs[self.OFFSET_OBS['PACKS'] + p * 3 + 1, self.OFFSET_TILE[self.curTile]] = 0

            self.shownTiles[self.curTile] -= 2
            self._history_emdedding_update()
            if p == 0:
                for i in range(2):
                    self.hand.append(self.curTile)
                self._hand_embedding_update()
            return
        if t[2] == 'Gang':
            self.packs[p].append(('GANG', self.curTile, (4 + p - self.tileFrom) % 4))

            self.obs[self.OFFSET_OBS['PACKS'] + p * 3 + 2, self.OFFSET_TILE[self.curTile]] = 1

            self.shownTiles[self.curTile] += 3 #新亮出了三张牌
            self._history_emdedding_update()
            if p == 0:
                for i in range(3):
                    self.hand.remove(self.curTile)
                self._hand_embedding_update()
                self.isAboutKong = True
            return
        if t[2] == 'AnGang':
            tile = 'CONCEALED' if p else t[3]
            self.packs[p].append(('GANG', tile, 0))
            if p == 0:
                self.isAboutKong = True

                self.obs[self.OFFSET_OBS['SELF_ANGANG'], self.OFFSET_TILE[tile]] = 1

                for i in range(4):
                    self.hand.remove(tile)
            else:
                self.isAboutKong = False
            return
        if t[2] == 'BuGang':
            tile = t[3]
            for i in range(len(self.packs[p])):
                if tile == self.packs[p][i][1]:
                    self.packs[p][i] = ('GANG', tile, self.packs[p][i][2])#把碰那个pack换成杠

                    self.obs[self.OFFSET_OBS['PACKS'] + p * 3 + 1, self.OFFSET_TILE[tile]] = 0
                    self.obs[self.OFFSET_OBS['PACKS'] + p * 3 + 2, self.OFFSET_TILE[tile]] = 1
                    
                    break
            self.shownTiles[tile] += 1
            self._history_emdedding_update()
            if p == 0:
                self.hand.remove(tile)
                self._hand_embedding_update()
                self.isAboutKong = True
                return
            else:
                # Available: Hu/Pass
                self.valid = []
                if self._check_mahjong(tile, isSelfDrawn = False, isAboutKong = True):
                    self.valid.append(self.OFFSET_ACT['Hu'])#抢杠胡
                self.valid.append(self.OFFSET_ACT['Pass'])
                return self._obs()
        raise NotImplementedError('Unknown request %s!' % request)
    
    '''
    Pass
    Hu
    Play XX
    Chi XX
    Peng
    Gang
    (An)Gang XX
    BuGang XX
    '''
    def action2response(self, action):
        if action < self.OFFSET_ACT['Hu']:
            return 'Pass'
        if action < self.OFFSET_ACT['Play']:
            return 'Hu'
        if action < self.OFFSET_ACT['Chi']:
            return 'Play ' + self.TILE_LIST[action - self.OFFSET_ACT['Play']]
        if action < self.OFFSET_ACT['Peng']:
            # 吃有63维，为3种花色 * 7种中心牌 * 3种位置
            t = (action - self.OFFSET_ACT['Chi']) // 3 #除以最后一维并取整，相当于去掉了最后一维位置信息
            return 'Chi ' + 'WTB'[t // 7] + str(t % 7 + 2) #再除以最后一维取整，相当于去掉了最后一维中心牌信息；除以最后一维取余，相当于保留了最后一维中心牌信息，加2是因为里边存储是用0~6表示两万到八万
        if action < self.OFFSET_ACT['Gang']:
            return 'Peng'
        if action < self.OFFSET_ACT['AnGang']:
            return 'Gang'
        if action < self.OFFSET_ACT['BuGang']:
            return 'Gang ' + self.TILE_LIST[action - self.OFFSET_ACT['AnGang']]
        return 'BuGang ' + self.TILE_LIST[action - self.OFFSET_ACT['BuGang']]
    
    '''
    Pass
    Hu
    Play XX
    Chi XX
    Peng
    Gang
    (An)Gang XX
    BuGang XX
    '''
    def response2action(self, response):
        t = response.split()
        if t[0] == 'Pass': return self.OFFSET_ACT['Pass']
        if t[0] == 'Hu': return self.OFFSET_ACT['Hu']
        if t[0] == 'Play': return self.OFFSET_ACT['Play'] + self.OFFSET_TILE[t[1]]
        if t[0] == 'Chi': return self.OFFSET_ACT['Chi'] + 'WTB'.index(t[1][0]) * 7 * 3 + (int(t[2][1]) - 2) * 3 + int(t[1][1]) - int(t[2][1]) + 1#中心牌的花色+数字+偏移
        if t[0] == 'Peng': return self.OFFSET_ACT['Peng'] + self.OFFSET_TILE[t[1]]
        if t[0] == 'Gang': return self.OFFSET_ACT['Gang'] + self.OFFSET_TILE[t[1]]
        if t[0] == 'AnGang': return self.OFFSET_ACT['AnGang'] + self.OFFSET_TILE[t[1]]
        if t[0] == 'BuGang': return self.OFFSET_ACT['BuGang'] + self.OFFSET_TILE[t[1]]
        return self.OFFSET_ACT['Pass']
    
    def _obs(self):
        mask = np.zeros(self.ACT_SIZE)
        for a in self.valid:
            mask[a] = 1
        return {
            'observation': self.obs.reshape((self.OBS_SIZE, 4, 9)).copy(),
            'action_mask': mask
        }
    #将手中的牌更新到状态矩阵里边
    def _hand_embedding_update(self):
        self.obs[self.OFFSET_OBS['HAND'] : self.OFFSET_OBS['HAND'] + 4] = 0
        d = defaultdict(int)
        for tile in self.hand:
            d[tile] += 1
        for tile in d:
            self.obs[self.OFFSET_OBS['HAND'] : self.OFFSET_OBS['HAND'] + d[tile], self.OFFSET_TILE[tile]] = 1
    
    def _history_emdedding_update(self):
        self.obs[self.OFFSET_OBS['REMAIN'] : self.OFFSET_OBS['REMAIN'] + 4] = 1
        for tile,counts in self.shownTiles.items():
            if counts != 0:
                self.obs[self.OFFSET_OBS['REMAIN'] : self.OFFSET_OBS['REMAIN'] + counts, self.OFFSET_TILE[tile]] = 0
        
        self.obs[self.OFFSET_OBS['PLAYED'] : self.OFFSET_OBS['PLAYED'] + 84] = 0

        for player in range(4):
            for t, tile in enumerate(self.history[player]):
                self.obs[self.OFFSET_OBS['PLAYED'] + player * 28 + t, self.OFFSET_TILE[tile]] = 1 

        



    def _check_mahjong(self, winTile, isSelfDrawn = False, isAboutKong = False):
        try:
            fans = MahjongFanCalculator(
                pack = tuple(self.packs[0]),
                hand = tuple(self.hand),
                winTile = winTile,
                flowerCount = 0,
                isSelfDrawn = isSelfDrawn,
                is4thTile = self.shownTiles[winTile] == 4,
                isAboutKong = isAboutKong,
                isWallLast = self.wallLast,
                seatWind = self.seatWind,
                prevalentWind = self.prevalentWind,
                verbose = True
            )
            fanCnt = 0
            for fanPoint, cnt, fanName, fanNameEn in fans:
                fanCnt += fanPoint * cnt
            if fanCnt < 8: raise Exception('Not Enough Fans')
        except:
            return False
        return True