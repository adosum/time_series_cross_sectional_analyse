from mytt.MyTT import *  # noqa: F403


def boduan(data):
    """
    布段函数 - 用于股票技术分析的综合指标计算

    data: pandas.DataFrame - 包含股票OHLC数据的DataFrame
    return: dict - 返回包含多个技术指标的字典，包括:
        - 主要指标: C1, C2 (威廉指标变种及其平滑版本)
        - 趋势指标: RSI5, ADX, DXR, PDI, MDI
        - 波动指标: TR1, WR10, bottom_stage, trends
        - 控盘指标: kong_pan, Q, Q1, tian, di
        - KDJ指标: K, D, RSV
        - 信号指标: alert, go, tiao, ready, kai_shi_la_sheng, qiang_la_sheng, jia_qiang_la_sheng
        - 辅助指标: VAR1, VAR2, AV, NEWVOL
    """
    # 提取基础价格数据
    CLOSE = data.close.values  # 收盘价
    HIGH = data.high.values  # 最高价
    LOW = data.low.values  # 最低价
    OPEN = data.open.values  # 开盘价

    # === RSI相关计算 ===
    LC = REF(CLOSE, 1)  # 前一日收盘价
    # RSI5: 5日相对强弱指标，衡量价格变化的强度
    RSI5 = (SMA(MAX((CLOSE - LC), 0), 5, 1) / SMA(ABS((CLOSE - LC)), 5, 1)) * 100

    # === DMI趋向指标相关计算 ===
    # TR1: 真实波幅的10日累计，衡量价格波动幅度
    TR1 = SUM(
        MAX(MAX((HIGH - LOW), ABS((HIGH - REF(CLOSE, 1)))), ABS((LOW - REF(CLOSE, 1)))),
        10,
    )
    HD = HIGH - REF(HIGH, 1)  # 最高价差值（今日最高价-昨日最高价）
    LD = REF(LOW, 1) - LOW  # 最低价差值（昨日最低价-今日最低价）
    DMP = SUM(IF(((HD > 0) & (HD > LD)), HD, 0), 10)  # 10日上升动向值
    DMM = SUM(IF(((LD > 0) & (LD > HD)), LD, 0), 10)  # 10日下降动向值
    PDI = (DMP * 100) / TR1  # 上升方向指标
    MDI = (DMM * 100) / TR1  # 下降方向指标
    ADX = MA(((ABS((MDI - PDI)) / (MDI + PDI)) * 100), 5)  # 趋向平均数，衡量趋势强度
    AV = RSI5 + ADX  # RSI和ADX的组合指标
    # DXR: 动向指数，结合当前和5日前的ADX值
    DXR = ((ADX + REF(ADX, 5)) / 2) + RSI5

    # === 威廉指标相关计算 ===
    # WR10: 威廉指标变种，衡量超买超卖状态
    WR10 = (100 * (HHV(HIGH, 10) - CLOSE)) / (HHV(HIGH, 13) - LLV(LOW, 13))
    NEWVOL = RSI5 - WR10  # RSI和威廉指标的差值
    bottom_stage = AV + NEWVOL  # 底部阶段指标，综合多个技术指标
    trends = LLV(bottom_stage, 1)  # 趋势指标，取底部阶段指标的最低值

    # === 强拉升指标计算 ===
    Y1 = LLV(LOW, 17)  # 17日内最低价
    Y2 = SMA(ABS(LOW - REF(LOW, 1)), 17, 1)  # 17日最低价变化幅度的平滑移动平均
    Y3 = SMA(MAX(LOW - REF(LOW, 1), 0), 17, 2)  # 17日最低价上涨幅度的平滑移动平均
    Q = -(EMA(IF(LOW <= Y1, Y2 / Y3, -3), 1))  # 强拉升准备指标

    # 强拉升信号：当Q指标穿越0轴时触发
    qiang_la_sheng = IF(CROSS(Q, 0), 1, 0)

    # === 加强拉升指标计算 ===
    # Q1: 价格相对于40日均线的偏离度百分比
    Q1 = (CLOSE - MA(CLOSE, 40)) / MA(CLOSE, 40) * 100

    # 加强拉升信号：当价格偏离度从-24%以下向上穿越时触发
    jia_qiang_la_sheng = IF(CROSS(Q1, -24), 1, 0)

    # === 控盘和拉升阶段指标 ===
    VAR1 = EMA(EMA(CLOSE, 9), 9)  # 双重指数移动平均，更平滑的价格趋势
    # 控盘指标：衡量价格变动的幅度（千分比）
    kong_pan = (VAR1 - REF(VAR1, 1)) / REF(VAR1, 1) * 1000

    # 三重指数移动平均的典型价格
    VAR2 = EMA(EMA(EMA((2 * CLOSE + HIGH + LOW) / 4, 4), 4), 4)
    # 天线：价格变动率的2日移动平均
    tian = MA((VAR2 - REF(VAR2, 1)) / REF(VAR2, 1) * 100, 2)
    # 地线：价格变动率的1日移动平均
    di = MA((VAR2 - REF(VAR2, 1)) / REF(VAR2, 1) * 100, 1)

    # 开始拉升信号：地线>天线且地线<0（底部反转信号）
    kai_shi_la_sheng = (di > tian) & (di < 0)

    # === KDJ随机指标计算 ===
    LOWV = LLV(LOW, 9)  # 9日内最低价
    HIGHV = HHV(HIGH, 9)  # 9日内最高价
    # RSV: 未成熟随机值，衡量收盘价在最近价格区间中的位置
    RSV = EMA((CLOSE - LOWV) / (HIGHV - LOWV) * 100, 3)
    K = EMA(RSV, 3)  # K值：RSV的指数移动平均
    D = MA(K, 3)  # D值：K值的移动平均

    # === 信号生成 ===
    alert = bottom_stage < 0  # 警报信号：底部阶段指标为负
    go = CROSS(bottom_stage, 0)  # 启动信号：底部阶段指标上穿0轴

    # === 威廉指标变种C1计算（主要返回值）===
    R1 = 1  # 比例因子
    R2 = (((2 * CLOSE) + HIGH) + LOW) / 4  # 典型价格的加权版本
    R4 = LLV(LOW, 5)  # 5日内最低价
    R5 = HHV(HIGH, 4)  # 4日内最高价
    # C1: 威廉指标的变种，衡量价格在近期区间中的相对位置
    C1 = EMA((((R2 - R4) / (R5 - R4)) * 100), 4) * R1
    # C2: C1的平滑版本，用于生成交叉信号
    C2 = EMA(((0.667 * REF(C1, 1)) + (0.333 * C1)), 2) * R1
    # 跳跃信号：C1上穿C2且C1<40（超卖区域的反弹信号）
    tiao = CROSS(C1, C2) & (C1 < 40)

    # === 综合准备信号计算 ===
    # 注释掉的控盘相关代码
    A10 = CROSS(kong_pan, 0)  # 控盘信号
    # 无庄控盘 = IF(kong_pan < 0, kong_pan, 0)  # 无庄控盘状态（暂未使用）
    开始控盘 = IF(A10, 20, 0)  # 开始控盘信号

    # AAZ2: 下影线长度超过1.4%的K线形态
    AAZ2 = (MIN(OPEN, CLOSE) - LOW) / LOW * 100 > 1.4
    # BBZ2: 价格接近30日均线且为阴线或十字星
    BBZ2 = (LOW / MA(CLOSE, 30) < 1.03) & (LOW / MA(CLOSE, 30) > 0.99) & (CLOSE <= OPEN)

    # 准备信号的两种计算方式（第二个会覆盖第一个）
    ready = IF(AAZ2 & BBZ2, 20, 0)  # 基于K线形态的准备信号
    ready = IF((tiao > 0) & (K > D) & CROSS(Q, 0), 20, 0)  # 综合技术指标的准备信号

    # 返回包含多个有用指标的字典
    return {
        # 主要指标
        "C1": C1,  # 威廉指标变种（主要指标）
        "C2": C2,  # C1的平滑版本
        # 趋势和动量指标
        "RSI5": RSI5,  # 5日相对强弱指标
        "ADX": ADX,  # 趋向平均数（趋势强度）
        "DXR": DXR,  # 动向指数
        "PDI": PDI,  # 上升方向指标
        "MDI": MDI,  # 下降方向指标
        # 波动性指标
        "TR1": TR1,  # 真实波幅累计
        "WR10": WR10,  # 威廉指标变种
        "bottom_stage": bottom_stage,  # 底部阶段指标
        "trends": trends,  # 趋势指标
        # 控盘和拉升指标
        "kong_pan": kong_pan,  # 控盘指标
        "Q": Q,  # 强拉升准备指标
        "Q1": Q1,  # 价格偏离度
        "tian": tian,  # 天线
        "di": di,  # 地线
        # KDJ随机指标
        "K": K,  # KDJ的K值
        "D": D,  # KDJ的D值
        "RSV": RSV,  # 未成熟随机值
        # 信号指标
        "alert": alert,  # 警报信号
        "go": go,  # 启动信号
        "tiao": tiao,  # 跳跃信号
        "ready": ready,  # 准备信号
        "kai_shi_la_sheng": kai_shi_la_sheng,  # 开始拉升信号
        "qiang_la_sheng": qiang_la_sheng,  # 强拉升信号
        "jia_qiang_la_sheng": jia_qiang_la_sheng,  # 加强拉升信号
        # 辅助指标
        "VAR1": VAR1,  # 双重指数移动平均
        "VAR2": VAR2,  # 三重指数移动平均典型价格
        "AV": AV,  # RSI和ADX组合指标
        "NEWVOL": NEWVOL,  # RSI和威廉指标差值
    }
