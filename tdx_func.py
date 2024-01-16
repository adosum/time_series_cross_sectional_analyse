from mytt.MyTT import *

def boduan(data):
    """
    data: np.array
    n: int
    return: np.array
    """
    CLOSE = data.close.values
    HIGH = data.high.values
    LOW = data.low.values
    OPEN = data.open.values

    LC = REF(CLOSE, 1)
    RSI5 = ((SMA(MAX((CLOSE - LC), 0), 5, 1) / SMA(ABS((CLOSE - LC)), 5, 1)) * 100)
    TR1 = SUM(MAX(MAX((HIGH - LOW), ABS((HIGH - REF(CLOSE, 1)))), ABS((LOW - REF(CLOSE, 1)))), 10)
    HD = (HIGH - REF(HIGH, 1))
    LD = (REF(LOW, 1) - LOW)
    DMP = SUM(IF(((HD > 0) & (HD > LD)), HD, 0), 10);
    DMM = SUM(IF(((LD > 0) & (LD > HD)), LD, 0), 10);
    PDI = ((DMP * 100) / TR1);
    MDI = ((DMM * 100) / TR1);
    ADX = MA(((ABS((MDI - PDI)) / (MDI + PDI)) * 100), 5);
    AV = (RSI5 + ADX);
    DXR = (((ADX + REF(ADX, 5)) / 2) + RSI5);

    WR10 = ((100 * (HHV(HIGH, 10) - CLOSE)) / (HHV(HIGH, 13) - LLV(LOW, 13)));
    NEWVOL = (RSI5 - WR10);
    bottom_stage = (AV + NEWVOL)
    trends = LLV(bottom_stage, 1)

    Y1 = LLV(LOW, 17)
    Y2 = SMA(ABS(LOW - REF(LOW, 1)), 17, 1)
    Y3 = SMA(MAX(LOW - REF(LOW, 1), 0), 17, 2)
    Q = -(EMA(IF(LOW <= Y1, Y2 / Y3, -3), 1))

    qiang_la_sheng = IF(CROSS(Q, 0), 1, 0)

    Q1 = (CLOSE - MA(CLOSE, 40)) / MA(CLOSE, 40) * 100

    jia_qiang_la_sheng = IF(CROSS(Q1, -24), 1, 0)

    VAR1 = EMA(EMA(CLOSE, 9), 9)
    kong_pan = (VAR1 - REF(VAR1, 1)) / REF(VAR1, 1) * 1000
    VAR2 = EMA(EMA(EMA((2 * CLOSE + HIGH + LOW) / 4, 4), 4), 4)
    tian = (MA((VAR2 - REF(VAR2, 1)) / REF(VAR2, 1) * 100, 2))
    di = (MA((VAR2 - REF(VAR2, 1)) / REF(VAR2, 1) * 100, 1))

    kai_shi_la_sheng = (di > tian) & (di < 0)

    LOWV = LLV(LOW, 9)
    HIGHV = HHV(HIGH, 9)
    RSV = EMA((CLOSE - LOWV) / (HIGHV - LOWV) * 100, 3)
    K = EMA(RSV, 3)
    D = MA(K, 3)

    alert = bottom_stage < 0
    go = CROSS(bottom_stage, 0);
    R1 = 1;
    R2 = ((((2 * CLOSE) + HIGH) + LOW) / 4);
    R4 = LLV(LOW, 5);
    R5 = HHV(HIGH, 4);
    C1 = (EMA((((R2 - R4) / (R5 - R4)) * 100), 4) * R1)
    C2 = (EMA(((0.667 * REF(C1, 1)) + (0.333 * C1)), 2) * R1)
    tiao = CROSS(C1, C2) & (C1 < 40)

    # A10=CROSS(kong_pan,0);
    # 无庄控盘 = IF(kong_pan<0,kong_pan,0)
    # 开始控盘 = IF(A10,20,0)
    AAZ2 = (MIN(OPEN, CLOSE) - LOW) / LOW * 100 > 1.4;
    BBZ2 = (LOW / MA(CLOSE, 30) < 1.03) & (LOW / MA(CLOSE, 30) > 0.99) & (CLOSE <= OPEN);
    ready = IF(AAZ2 & BBZ2, 20, 0);
    ready = IF((tiao > 0) & (K > D) & CROSS(Q, 0), 20, 0)
    return C1
