import pandas as pd


# Strict Chan Theory Analysis Class (Two-stage inclusion compression + Stroke direction linkage + Strict segment division)
class ChanAnalysis:
    def __init__(self, df):
        self.raw_df = df.copy()
        self.df = df.copy()
        self.df["index"] = range(len(self.df))
        # Stage 1: Uncertain direction, use outer inclusion logic for initial compression
        self.comp_df = self._compress_inclusion(self.df, direction=None)
        self.first_stroke_direction = None  # First stroke direction
        self._macd_cached = None  # Cache compressed K-line MACD results

    # === Basic inclusion compression ===
    def _compress_inclusion(self, df, direction=None):
        bars = []
        for i in range(len(df)):
            bar = df.iloc[i]
            high = bar["high"]
            low = bar["low"]
            date = bar["date"]
            if not bars:
                bars.append(
                    {
                        "index": bar["index"],
                        "date": date,
                        "open": bar["open"],
                        "close": bar["close"],
                        "high": high,
                        "low": low,
                    }
                )
                continue
            prev = bars[-1]
            prev_high, prev_low = prev["high"], prev["low"]
            inclusion = (high <= prev_high and low >= prev_low) or (
                high >= prev_high and low <= prev_low
            )
            if inclusion:
                if direction is None:  # Simple outer inclusion
                    prev["high"] = max(prev_high, high)
                    prev["low"] = min(prev_low, low)
                    prev["close"] = bar["close"]
                    prev["date"] = date
                elif direction == "up":
                    prev["high"] = max(prev_high, high)
                    prev["close"] = bar["close"]
                    prev["date"] = date
                else:  # down
                    prev["low"] = min(prev_low, low)
                    prev["close"] = bar["close"]
                    prev["date"] = date
            else:
                bars.append(
                    {
                        "index": bar["index"],
                        "date": date,
                        "open": bar["open"],
                        "close": bar["close"],
                        "high": high,
                        "low": low,
                    }
                )
        return pd.DataFrame(bars)

    # === MACD ===
    def compute_macd(self, fast=12, slow=26, signal=9, price_col="close"):
        if self._macd_cached is not None and len(self._macd_cached) == len(
            self.comp_df
        ):
            return self._macd_cached
        df = self.comp_df.copy()
        close = df[price_col]
        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()
        dif = ema_fast - ema_slow
        dea = dif.ewm(span=signal, adjust=False).mean()
        hist = dif - dea
        macd_df = pd.DataFrame(
            {
                "index": df["index"].values,
                "DIF": dif.values,
                "DEA": dea.values,
                "HIST": hist.values,
                "close": df["close"].values,
            }
        )
        self._macd_cached = macd_df
        return macd_df

    def _segment_macd_stats(self, segments, macd_df):
        if not segments:
            return {}
        macd_map = macd_df.set_index("index")
        stats = {}
        for idx, seg in enumerate(segments):
            start_i = seg["start_index"]
            end_i = seg["end_index"]
            sub = macd_map.loc[macd_map.index.intersection(range(start_i, end_i + 1))]
            if sub.empty:
                continue
            hist = sub["HIST"]
            stats[idx] = {
                "area_signed": hist.sum(),
                "area_abs": hist.abs().sum(),
                "peak": hist.max() if seg["direction"] == "up" else hist.min(),
                "avg": hist.mean(),
                "direction": seg["direction"],
            }
        return stats

    def _detect_macd_divergence(self, segments, macd_stats, lookback=3):
        pts = []
        for i in range(2, len(segments)):
            s1, s2, s3 = segments[i - 2], segments[i - 1], segments[i]
            if s1["direction"] == s2["direction"] == s3["direction"]:
                d = s1["direction"]
                st1 = macd_stats.get(i - 2)
                st2 = macd_stats.get(i - 1)
                st3 = macd_stats.get(i)
                if not (st1 and st2 and st3):
                    continue
                price_move_ok = (
                    d == "up" and s3["end_price"] >= s2["end_price"] >= s1["end_price"]
                ) or (
                    d == "down"
                    and s3["end_price"] <= s2["end_price"] <= s1["end_price"]
                )
                if not price_move_ok:
                    continue
                if d == "up":
                    cond_peak = st3["peak"] < st2["peak"] < st1["peak"]
                    cond_area = st3["area_abs"] < st2["area_abs"] < st1["area_abs"]
                else:
                    cond_peak = (
                        st3["peak"] > st2["peak"] > st1["peak"]
                    )  # Negative value rising
                    cond_area = st3["area_abs"] < st2["area_abs"] < st1["area_abs"]
                if cond_peak and cond_area:
                    pts.append(
                        {
                            "index": s3["end_index"],
                            "date": s3["end_date"],
                            "type": "warn",
                            "pattern": "beichi_macd",
                            "price": s3["end_price"],
                            "description": "MACD area and peak decreasing divergence",
                        }
                    )
        return pts

    # === Fenxing / Bi / Xianduan ===
    def find_fractal(self, df_override=None):
        df = (df_override if df_override is not None else self.comp_df).copy()
        raw = []
        for i in range(1, len(df) - 1):
            mid = df.iloc[i]
            prev = df.iloc[i - 1]
            nxt = df.iloc[i + 1]
            if mid["high"] > prev["high"] and mid["high"] > nxt["high"]:
                raw.append(
                    {
                        "index": mid["index"],
                        "date": mid["date"],
                        "type": "top",
                        "price": mid["high"],
                    }
                )
            if mid["low"] < prev["low"] and mid["low"] < nxt["low"]:
                raw.append(
                    {
                        "index": mid["index"],
                        "date": mid["date"],
                        "type": "bottom",
                        "price": mid["low"],
                    }
                )
        filtered = []
        for f in raw:
            if not filtered:
                filtered.append(f)
                continue
            last = filtered[-1]
            if f["type"] == last["type"]:
                if (f["type"] == "top" and f["price"] > last["price"]) or (
                    f["type"] == "bottom" and f["price"] < last["price"]
                ):
                    filtered[-1] = f
                else:
                    continue
            else:
                filtered.append(f)
        return filtered

    def find_strokes(self, fractals, df_override=None):
        if len(fractals) < 2:
            return []
        df = (df_override if df_override is not None else self.comp_df).set_index(
            "index"
        )
        strokes = []
        i = 0
        while i < len(fractals) - 1:
            a = fractals[i]
            b = fractals[i + 1]
            if a["type"] == b["type"]:
                i += 1
                continue
            if b["index"] - a["index"] < 2:
                i += 1
                continue
            middle = [idx for idx in df.index if a["index"] < idx < b["index"]]
            valid = True
            if a["type"] == "top" and b["type"] == "bottom":
                for idx in middle:
                    row = df.loc[idx]
                    if not (row["high"] < a["price"] and row["low"] > b["price"]):
                        valid = False
                        break
            elif a["type"] == "bottom" and b["type"] == "top":
                for idx in middle:
                    row = df.loc[idx]
                    if not (row["low"] > a["price"] and row["high"] < b["price"]):
                        valid = False
                        break
            if valid:
                direction = "up" if b["price"] > a["price"] else "down"
                strokes.append(
                    {
                        "start_index": a["index"],
                        "end_index": b["index"],
                        "start_date": a["date"],
                        "end_date": b["date"],
                        "start_price": a["price"],
                        "end_price": b["price"],
                        "direction": direction,
                    }
                )
                if self.first_stroke_direction is None:
                    self.first_stroke_direction = direction
            i += 1
        return strokes

    def recompress_after_first_stroke(self):
        if self.first_stroke_direction is None:
            return
        self.comp_df = self._compress_inclusion(
            self.df, direction=self.first_stroke_direction
        )
        self._macd_cached = None

    def _stroke_range(self, stroke):
        low = min(stroke["start_price"], stroke["end_price"])
        high = max(stroke["start_price"], stroke["end_price"])
        return low, high

    def find_segments(self, strokes, min_strokes=3, overlap_lookback=3):
        if not strokes:
            return []
        segments = []
        n = len(strokes)
        i = 0
        while i < n:
            if i + 2 >= n:
                break
            candidate = [strokes[i], strokes[i + 1], strokes[i + 2]]
            j = i + 3
            terminated = False
            while j < n:
                next_stroke = strokes[j]
                direction = (
                    "up"
                    if candidate[-1]["end_price"] > candidate[0]["start_price"]
                    else "down"
                )
                next_low, next_high = self._stroke_range(next_stroke)
                overlap = False
                lookback_strokes = candidate[: min(len(candidate), overlap_lookback)]
                for early in lookback_strokes:
                    el, eh = self._stroke_range(early)
                    if not (next_high < el or next_low > eh):
                        overlap = True
                        break
                if overlap:
                    seg_direction = (
                        "up"
                        if candidate[-1]["end_price"] > candidate[0]["start_price"]
                        else "down"
                    )
                    segments.append(
                        self._build_segment(
                            candidate, direction=seg_direction, reason="failure_overlap"
                        )
                    )
                    candidate = [candidate[-1], next_stroke]
                    j += 1
                    while j < n and len(candidate) < min_strokes:
                        candidate.append(strokes[j])
                        j += 1
                    terminated = True
                    continue
                else:
                    candidate.append(next_stroke)
                    j += 1
            if candidate and len(candidate) >= min_strokes:
                seg_direction = (
                    "up"
                    if candidate[-1]["end_price"] > candidate[0]["start_price"]
                    else "down"
                )
                reason = "end_of_data" if not terminated else "extended_end"
                segments.append(
                    self._build_segment(
                        candidate, direction=seg_direction, reason=reason
                    )
                )
            last_seg_last = segments[-1]["strokes"][-1]
            last_idx_all = strokes.index(last_seg_last)
            if last_idx_all >= n - 2:
                break
            i = max(last_idx_all - 1, 0)
        return segments

    def _build_segment(self, strokes, direction=None, reason=""):
        if direction is None:
            direction = (
                "up" if strokes[-1]["end_price"] > strokes[0]["start_price"] else "down"
            )
        return {
            "start_index": strokes[0]["start_index"],
            "end_index": strokes[-1]["end_index"],
            "start_date": strokes[0]["start_date"],
            "end_date": strokes[-1]["end_date"],
            "start_price": strokes[0]["start_price"],
            "end_price": strokes[-1]["end_price"],
            "direction": direction,
            "strokes_count": len(strokes),
            "strokes": strokes,
            "termination": reason,
        }

    def find_zhongshu(self, strokes):
        if len(strokes) < 3:
            return []
        zlist = []

        def hi_lo(p):
            return max(p["start_price"], p["end_price"]), min(
                p["start_price"], p["end_price"]
            )

        for i in range(len(strokes) - 2):
            s1, s2, s3 = strokes[i], strokes[i + 1], strokes[i + 2]
            highs = [hi_lo(s)[0] for s in (s1, s2, s3)]
            lows = [hi_lo(s)[1] for s in (s1, s2, s3)]
            upper = min(highs)
            lower = max(lows)
            if upper > lower:
                zlist.append(
                    {
                        "start_index": s1["start_index"],
                        "end_index": s3["end_index"],
                        "upper": upper,
                        "lower": lower,
                        "center": (upper + lower) / 2,
                    }
                )
        merged = []
        for zs in zlist:
            if not merged:
                merged.append(zs)
                continue
            last = merged[-1]
            if zs["lower"] <= last["upper"] and zs["upper"] >= last["lower"]:
                nu = min(last["upper"], zs["upper"])
                nl = max(last["lower"], zs["lower"])
                last.update(
                    {
                        "upper": nu,
                        "lower": nl,
                        "end_index": zs["end_index"],
                        "center": (nu + nl) / 2,
                    }
                )
            else:
                merged.append(zs)
        return merged

    def _get_segment_extrema(self, segment):
        """
        Helper function: Get the true highest and lowest prices of a segment within all its strokes
        """
        if not segment or not segment.get("strokes"):
            return segment.get("start_price", 0), segment.get("end_price", 0)

        all_prices = []
        for s in segment["strokes"]:
            all_prices.append(s["start_price"])
            all_prices.append(s["end_price"])

        if not all_prices:
            return segment["start_price"], segment["end_price"]

        return min(all_prices), max(all_prices)

    def find_buy_sell_points(self, segments, zhongshu):
        pts = []

        # --- Yi mai / Er mai / Yi mai / Er mai ---
        for i in range(1, len(segments)):
            ps = segments[i - 1]  # Previous segment
            s = segments[i]  # Current segment

            # 1. Yi mai and Er mai judgment
            if ps["direction"] == "down" and s["direction"] == "up":
                # Yi mai: Starting point of segment reversal
                yi_mai_low = ps["end_price"]
                pts.append(
                    {
                        "index": s["start_index"],
                        "date": s["start_date"],
                        "type": "buy",
                        "pattern": "yi_mai",
                        "price": s["start_price"],
                        "description": "Segment reversal yi mai",
                    }
                )

                # Er mai: Next down segment after yi mai (pullback segment)
                if i + 1 < len(segments):
                    pullback = segments[i + 1]
                    if pullback["direction"] == "down":
                        # Get the true low point of pullback segment
                        pull_low, _ = self._get_segment_extrema(
                            pullback
                        )  # Need helper function

                        # Core condition: pullback low > yi mai low
                        if pull_low > yi_mai_low:
                            pts.append(
                                {
                                    "index": pullback["end_index"],
                                    "date": pullback["end_date"],
                                    "type": "buy",
                                    "pattern": "er_mai",
                                    "price": pullback["end_price"],
                                    "description": "Er mai: Pullback after yi mai did not make new low",
                                }
                            )

            # 2. Yi mai and Er mai judgment
            if ps["direction"] == "up" and s["direction"] == "down":
                # Yi mai: Starting point of segment reversal
                yi_mai_high = ps["end_price"]
                pts.append(
                    {
                        "index": s["start_index"],
                        "date": s["start_date"],
                        "type": "sell",
                        "pattern": "yi_mai",
                        "price": s["start_price"],
                        "description": "Segment reversal yi mai",
                    }
                )

                # Er mai: Next up segment after yi mai (pullback segment)
                if i + 1 < len(segments):
                    pullback = segments[i + 1]
                    if pullback["direction"] == "up":
                        # Get the true high point of pullback segment
                        _, pull_high = self._get_segment_extrema(
                            pullback
                        )  # Need helper function

                        # Core condition: pullback high < yi mai high
                        if pull_high < yi_mai_high:
                            pts.append(
                                {
                                    "index": pullback["end_index"],
                                    "date": pullback["end_date"],
                                    "type": "sell",
                                    "pattern": "er_mai",
                                    "price": pullback["end_price"],
                                    "description": "Er mai: Pullback after yi mai did not make new high",
                                }
                            )

        # --- Segment same direction decreasing divergence warning (keep original logic) ---
        for i in range(2, len(segments)):
            s1, s2, s3 = segments[i - 2], segments[i - 1], segments[i]
            if s1["direction"] == s2["direction"] == s3["direction"]:
                amp = lambda seg: abs(seg["end_price"] - seg["start_price"])  # noqa: E731
                if amp(s3) < amp(s2) < amp(s1):
                    pts.append(
                        {
                            "index": s3["end_index"],
                            "date": s3["end_date"],
                            "type": "warn",
                            "pattern": "beichi",
                            "price": s3["end_price"],
                            "description": "Segment same direction decreasing divergence warning",
                        }
                    )

        return pts

    def find_advanced_buy_sell_points(
        self,
        segments,
        zhongshu,
        dist_threshold=0.03,
        dist2_threshold=None,
        tiny_penetration=0.01,
        debug=False,
    ):
        """
        Advanced buy/sell points (corrected version):
        1. Fixed the concept confusion of "er mai/san mai", here judging is zhongshu related "san mai/san mai".
        2. Fixed the BUG of getting pullback segment (pull) lowest/highest point.
        3. Apply penetration (tiny_penetration) logic to san mai/san mai to relax strict conditions.
        """
        if dist2_threshold is None:
            dist2_threshold = dist_threshold

        # !! Note: your original find_buy_sell_points already contains "yi_mai" (yi mai)
        # !! The real "er_mai" (er mai) logic should be supplemented there (i.e., first pullback after yi mai does not make new low)
        # !! Here focus on correcting your original zhongshu buy/sell points (i.e., san mai)
        points_basic = self.find_buy_sell_points(segments, zhongshu)

        advanced = []
        debug_buy_fail = []
        debug_sell_fail = []

        for seg_idx, seg in enumerate(segments):
            related = [
                zs
                for zs in zhongshu
                if not (
                    zs["end_index"] < seg["start_index"]
                    or zs["start_index"] > seg["end_index"]
                )
            ]
            if not related:
                continue

            for zs in related:
                up, low = zs["upper"], zs["lower"]

                # ========= Buy side (san mai: upward breakthrough of upper edge) =========
                if seg["direction"] == "up":
                    # Confirm breakthrough: segment starts from inside/lower edge of zhongshu, breaks upward through upper edge
                    if seg["start_price"] <= up and seg["end_price"] > up:
                        if seg_idx + 1 < len(segments):
                            pull = segments[seg_idx + 1]
                            if pull["direction"] == "down":
                                # --- Fix point 1: Get segment's true lowest point ---
                                # OLD: pull_low = min(pull['start_price'], pull['end_price'])
                                pull_low, _ = self._get_segment_extrema(pull)  # NEW

                                # --- Fix point 2: Apply penetration logic to san mai judgment ---
                                penetration_limit = up * (1 - tiny_penetration)
                                penetration_ok = pull_low >= penetration_limit

                                # --- Fix point 3: Distance calculation uses abs, relies on penetration_ok judgment ---
                                dist = abs(pull_low - up) / up

                                # OLD: if pull_low >= up and 0 <= dist < dist_threshold:
                                if penetration_ok and dist < dist_threshold:  # NEW
                                    # --- Fix point 4: Concept correction, this is "san mai" ---
                                    advanced.append(
                                        {
                                            "index": pull["end_index"],
                                            "date": pull["end_date"],
                                            "type": "buy",
                                            "pattern": "san_mai",
                                            "price": pull[
                                                "end_price"
                                            ],  # Signal point at end of pullback
                                            "description": f"San mai: Pullback near upper edge did not deeply break (dist {dist:.3f})",
                                        }
                                    )

                                    # --- San mai confirmation (original san_mai logic) ---
                                    if seg_idx + 2 < len(segments):
                                        up2 = segments[seg_idx + 2]
                                        # Continuing upward segment must make new high
                                        if (
                                            up2["direction"] == "up"
                                            and up2["end_price"] > seg["end_price"]
                                            and seg_idx + 3 < len(segments)
                                        ):
                                            pull2 = segments[seg_idx + 3]
                                            if pull2["direction"] == "down":
                                                # Fix point 1 (same application)
                                                pull2_low, _ = (
                                                    self._get_segment_extrema(pull2)
                                                )

                                                penetration_limit_2 = up * (
                                                    1 - tiny_penetration
                                                )
                                                penetration_ok_2 = (
                                                    pull2_low >= penetration_limit_2
                                                )
                                                dist2 = abs(pull2_low - up) / up

                                                if (
                                                    penetration_ok_2
                                                    and dist2 < dist2_threshold
                                                ):
                                                    advanced.append(
                                                        {
                                                            "index": pull2["end_index"],
                                                            "date": pull2["end_date"],
                                                            "type": "buy",
                                                            "pattern": "san_mai_confirm",
                                                            "price": pull2["end_price"],
                                                            "description": f"San mai confirmation: Second pullback dist from upper edge {dist2:.3f}",
                                                        }
                                                    )
                                                else:
                                                    if debug:
                                                        debug_buy_fail.append(
                                                            {
                                                                "seg_break": seg_idx,
                                                                "seg_extend": seg_idx
                                                                + 2,
                                                                "seg_pull2": seg_idx
                                                                + 3,
                                                                "up": up,
                                                                "pull2_low": pull2_low,
                                                                "penetration_ok": penetration_ok_2,
                                                                "dist2": dist2,
                                                            }
                                                        )

                # ========= Sell side (san mai: downward breakthrough of lower edge) =========
                if seg["direction"] == "down":
                    # Confirm breakdown: segment starts from inside/upper edge of zhongshu, breaks downward through lower edge
                    if seg["start_price"] >= low and seg["end_price"] < low:
                        if seg_idx + 1 < len(segments):
                            pull = segments[seg_idx + 1]
                            if pull["direction"] == "up":
                                # --- Fix point 1: Get segment's true highest point ---
                                # OLD: pull_high = max(pull['start_price'], pull['end_price'])
                                _, pull_high = self._get_segment_extrema(pull)  # NEW

                                # --- Fix point 2: Apply penetration logic ---
                                penetration_limit = low * (1 + tiny_penetration)
                                penetration_ok = pull_high <= penetration_limit

                                # --- Fix point 3: Distance calculation uses abs ---
                                dist = abs(pull_high - low) / low

                                # OLD: if pull_high <= low and 0 <= dist < dist_threshold:
                                if penetration_ok and dist < dist_threshold:  # NEW
                                    # --- Fix point 4: Concept correction, this is "san mai" ---
                                    advanced.append(
                                        {
                                            "index": pull["end_index"],
                                            "date": pull["end_date"],
                                            "type": "sell",
                                            "pattern": "san_mai",
                                            "price": pull["end_price"],
                                            "description": f"San mai: Pullback near lower edge did not strongly penetrate (dist {dist:.3f})",
                                        }
                                    )

                                    # --- San mai confirmation (original san_mai logic) ---
                                    if seg_idx + 2 < len(segments):
                                        dn2 = segments[seg_idx + 2]
                                        # Continuing downward segment must make new low
                                        if (
                                            dn2["direction"] == "down"
                                            and dn2["end_price"] < seg["end_price"]
                                            and seg_idx + 3 < len(segments)
                                        ):
                                            pull2 = segments[seg_idx + 3]
                                            if pull2["direction"] == "up":
                                                # Fix point 1 (same application)
                                                _, pull2_high = (
                                                    self._get_segment_extrema(pull2)
                                                )

                                                penetration_limit_2 = low * (
                                                    1 + tiny_penetration
                                                )
                                                penetration_ok_2 = (
                                                    pull2_high <= penetration_limit_2
                                                )
                                                dist2 = abs(pull2_high - low) / low

                                                if (
                                                    penetration_ok_2
                                                    and dist2 < dist2_threshold
                                                ):
                                                    advanced.append(
                                                        {
                                                            "index": pull2["end_index"],
                                                            "date": pull2["end_date"],
                                                            "type": "sell",
                                                            "pattern": "san_mai_confirm",
                                                            "price": pull2["end_price"],
                                                            "description": f"San mai confirmation: Second pullback dist from lower edge {dist2:.3f}",
                                                        }
                                                    )
                                                else:
                                                    if debug:
                                                        debug_sell_fail.append(
                                                            {
                                                                "seg_break": seg_idx,
                                                                "seg_extend": seg_idx
                                                                + 2,
                                                                "seg_pull2": seg_idx
                                                                + 3,
                                                                "low": low,
                                                                "pull2_high": pull2_high,
                                                                "penetration_ok": penetration_ok_2,
                                                                "dist2": dist2,
                                                            }
                                                        )

        # --- (Subsequent MACD and deduplication logic remains unchanged) ---
        macd_df = self.compute_macd()
        macd_stats = self._segment_macd_stats(segments, macd_df)
        diver = self._detect_macd_divergence(segments, macd_stats)

        all_pts = points_basic + advanced + diver
        unique = {}
        for p in all_pts:
            key = (p["index"], p["pattern"])
            if key not in unique:
                unique[key] = p

        result = list(unique.values())

        if debug and (debug_buy_fail or debug_sell_fail):
            print(
                "=== San mai/San mai candidates failed pullback distance filtering statistics ==="
            )

            def _print_fail(lst, title):
                if not lst:
                    return
                print(f"{title} failed: {len(lst)}")
                for item in lst[:10]:
                    print(item)
                if lst:  # Avoid division by zero error
                    dists = [item["dist2"] for item in lst]
                    print(
                        f"{title} dist2 distribution: min={min(dists):.4f} max={max(dists):.4f} avg={sum(dists) / len(dists):.4f}"
                    )

            _print_fail(debug_buy_fail, "San mai confirmation")
            _print_fail(debug_sell_fail, "San mai confirmation")

        return result
