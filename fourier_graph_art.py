# -*- coding: utf-8 -*-
"""
フーリエ級数（エピサイクル）による「関数グラフアート」生成プログラム
=================================================================

仕組み:
  1. 画像から輪郭点を抽出し、複素数列 z_n = x_n + i*y_n に変換する
  2. z_n を離散フーリエ変換(DFT)して、各周波数 k ごとの
     回転ベクトル(エピサイクル)の半径・初期位相を求める
  3. 半径の大きい順に並べ替え、時刻 t=0..N-1 で
     逆変換(=ベクトルの合成)を行いながら、その軌跡をアニメーション表示する

必要ライブラリ:
    pip install numpy opencv-python matplotlib scikit-image scipy --break-system-packages

使い方:
    python fourier_graph_art.py --image path/to/your_image.png --out output.gif

  --image           読み込む画像ファイルのパス（線画・シルエットが最も綺麗に出ます）
  --out             出力アニメーションファイル（.gif または .mp4）
  --n_points        パスから間引くサンプル点数（多いほど精細、既定 800）
  --n_circles       描画に使うエピサイクルの数（既定: 全部）
  --frames          1周期分のアニメーションフレーム数（既定 = n_points）
  --fps             アニメーションのフレームレート（既定 30）
  --mode            skeleton(既定): 線画を1px幅の中心線に細めてから辿る（二重線にならず綺麗）
                     edge: Cannyで検出した輪郭の縁を辿る(旧方式) / silhouette: 塗りつぶし外形のみ
  --canny1 --canny2 edgeモードでのCannyしきい値
  --min_contour_len 採用するパーツの最小点数（小さいノイズの除去用）
  --max_dim         処理前に画像の長辺をこのピクセル数に縮小(既定 600。大きすぎる画像の高速化・ノイズ低減用)

補足:
  目・口などの「別パーツ」も1本の連続パスに繋げて描画するため、
  各パーツを描き終えると次のパーツへ移動する短いジャンプ線が入ります
  (これは静止画の「クリーン版」で自動的に消せます)。
  skeletonモードでは各パーツ自体は分岐を持つ線画でも1本の中心線として
  たどるため、旧方式のような二重線・ガタつきが大幅に減ります。
"""

import argparse
import numpy as np
import cv2
from skimage.morphology import skeletonize
from scipy.ndimage import label as cc_label
import matplotlib
matplotlib.use("Agg")  # サーバ環境でも動くように（GUI不要のバックエンド）
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


# ----------------------------------------------------------------------
# 1. 画像 → 輪郭点(複素数列) への変換
# ----------------------------------------------------------------------
def image_to_complex_points(image_path: str, n_points: int = 300,
                             mode: str = "skeleton",
                             canny1: int = 50, canny2: int = 150,
                             min_contour_len: int = 15,
                             max_dim: int = 600):
    """
    画像を読み込み、線画のパーツ（外形・目・口・模様など）をすべて抽出して
    1本の連続パスに繋げたうえで、等間隔に n_points 個サンプリングした
    複素数列 z_n を返す。

    mode:
      "skeleton"   線を1px幅の中心線に細めてから、分岐も含めて1本の線として
                   辿る。二重線にならず、なめらかな結果になる（既定）
      "edge"       Cannyエッジ検出で見つけた輪郭の縁をそのまま辿る(旧方式)
      "silhouette" 大津の二値化を使う（塗りつぶしシルエットのみ欲しい場合向け）

    戻り値:
      z         複素数列 (n_points,)
      jump_mask 真偽値配列 (n_points,)。True の点は「パーツ間の移動線
                (輪郭→目→口 のような継ぎ目)」上にある点であることを示す。
                これを使うと、あとで継ぎ目の線だけを消した「クリーン版」を描ける。
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"画像が読み込めません: {image_path}")

    # 大きすぎる画像は処理速度・ノイズ低減のため縮小
    h, w = img.shape
    if max(h, w) > max_dim:
        scale_f = max_dim / max(h, w)
        img = cv2.resize(img, (int(w * scale_f), int(h * scale_f)),
                          interpolation=cv2.INTER_AREA)

    blurred = cv2.GaussianBlur(img, (3, 3), 0)

    if mode == "skeleton":
        ordered = _extract_pieces_skeleton(blurred, min_contour_len)
    else:
        ordered = _extract_pieces_contour(blurred, mode, canny1, canny2, min_contour_len)

    if not ordered:
        raise ValueError("有効なパーツがありませんでした。しきい値やモードを見直してください。")

    # 各パーツごとに、次のパーツへ移動する1本の連続点列にする。
    # 同時に、各点がどのパーツに属するかを piece_ids に記録する。
    pieces = []
    piece_ids = []
    for i, c in enumerate(ordered):
        pieces.append(c.astype(float))
        piece_ids.extend([i] * len(c))
        if not _is_closed_path(c):
            # 輪郭ベースのパーツは始点に戻って自身のループを閉じる
            pieces.append(c[:1].astype(float))
            piece_ids.append(i)
    points = np.concatenate(pieces, axis=0)
    piece_ids = np.array(piece_ids)

    # 隣り合う点が別パーツに属する区間 = パーツ間の移動線(ジャンプ)
    next_ids = np.roll(piece_ids, -1)  # 末尾は先頭(=全体を閉じる区間)と比較
    is_jump_segment = piece_ids != next_ids

    xs = points[:, 0]
    ys = points[:, 1]

    # 画像は y が下向きなので、見た目を自然にするため反転
    ys = -ys

    # 重心を原点に、スケールを正規化
    xs = xs - xs.mean()
    ys = ys - ys.mean()
    scale = max(xs.max() - xs.min(), ys.max() - ys.min())
    if scale > 0:
        xs = xs / scale * 400
        ys = ys / scale * 400

    z = xs + 1j * ys

    # 全体を弧長ベースで等間隔に n_points 個へリサンプリングしつつ、
    # 各リサンプリング点がジャンプ区間上にあるかどうかも一緒に求める
    z, jump_mask = resample_by_arclength(z, n_points, is_jump_segment)
    return z, jump_mask


def _is_closed_path(c: np.ndarray) -> bool:
    """パスの始点と終点が(ほぼ)同じ座標かどうか。skeletonモードのDFSパスは自然に閉じる。"""
    return len(c) > 2 and np.allclose(c[0], c[-1])


# ----------------------------------------------------------------------
# 1a. 旧方式: Canny/二値化 + findContours で輪郭の縁を辿る
# ----------------------------------------------------------------------
def _extract_pieces_contour(blurred, mode, canny1, canny2, min_contour_len):
    if mode == "silhouette":
        _, mask = cv2.threshold(blurred, 0, 255,
                                 cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
        mask = cv2.Canny(blurred, canny1, canny2)
        kernel = np.ones((2, 2), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError("輪郭が見つかりませんでした。画像のコントラストや --mode を確認してください。")

    contours = [c.squeeze(axis=1) if c.ndim == 3 else c.reshape(-1, 2)
                for c in contours if len(c) >= min_contour_len]
    if not contours:
        return []

    contours.sort(key=len, reverse=True)
    return _order_contours_greedy(contours)


# ----------------------------------------------------------------------
# 1b. 新方式: スケルトン化 + グラフ上のDFSで中心線を1本の線として辿る
# ----------------------------------------------------------------------
def _extract_pieces_skeleton(blurred, min_contour_len,
                              adaptive_block=21, adaptive_c=8):
    """
    線画を1px幅の中心線(スケルトン)に細め、連結成分ごとにグラフを作って
    DFS(深さ優先探索)で全エッジを辿るパスを作る。
    分岐(交差点)では、行き止まりに達すると同じ道を引き返す(バックトラック)ため、
    引き返した部分は元の線とぴったり重なって描かれ、二重線には見えない。

    "インク"の検出には局所適応二値化(adaptiveThreshold)を使う。
    これは周囲より局所的に暗い画素だけを拾うため、
      - 単純な明暗二値化(Otsu)だと、色付きイラストの塗りつぶし部分ごと
        "インク"とみなされて目・鼻・ベルトなどの細部が欠落する
      - Cannyエッジ検出だと、太い線の場合に線の両端(2本のエッジ)を
        別々に拾ってしまい、スケルトン化しても二重線が残る
    という、それぞれの弱点を避けられる。太い線でもストローク全体が
    "インク"として1つのまとまりで検出されるので、スケルトン化すると
    綺麗に1本の中心線になる。
    """
    mask = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
        adaptive_block, adaptive_c)
    mask_bool = mask > 0

    skeleton = skeletonize(mask_bool)

    labeled, n_labels = cc_label(skeleton, structure=np.ones((3, 3)))
    pieces = []
    for label_id in range(1, n_labels + 1):
        ys, xs = np.nonzero(labeled == label_id)
        if len(ys) < min_contour_len:
            continue
        coords = list(zip(ys.tolist(), xs.tolist()))
        path_yx = _skeleton_component_to_path(coords)
        # (y, x) -> (x, y) のピクセル座標に変換
        path_xy = np.array([[x, y] for (y, x) in path_yx], dtype=float)
        pieces.append(path_xy)

    if not pieces:
        return []

    pieces.sort(key=len, reverse=True)
    return _order_contours_greedy(pieces)


def _skeleton_component_to_path(coords):
    """
    スケルトン画素の集合(1つの連結成分)を、8近傍で繋がったグラフとみなし、
    DFS+バックトラックで全エッジを辿る1本のパス(座標列)を作って返す。
    """
    coord_set = set(coords)
    neighbors = {}
    for (y, x) in coords:
        nb = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                n = (y + dy, x + dx)
                if n in coord_set:
                    nb.append(n)
        neighbors[(y, x)] = nb

    # 端点(次数1)があればそこから開始した方が自然な描き出しになる
    start = next((c for c in coords if len(neighbors[c]) == 1), coords[0])

    def edge_key(a, b):
        return (a, b) if a <= b else (b, a)

    visited_edges = set()
    path = [start]
    trav_stack = [start]

    while trav_stack:
        current = trav_stack[-1]
        next_node = None
        for n in neighbors[current]:
            ek = edge_key(current, n)
            if ek not in visited_edges:
                next_node = n
                break
        if next_node is not None:
            visited_edges.add(edge_key(current, next_node))
            trav_stack.append(next_node)
            path.append(next_node)
        else:
            trav_stack.pop()
            if trav_stack:
                path.append(trav_stack[-1])  # 既に描いた道をそのまま引き返す

    return path


def _order_contours_greedy(contours):
    """
    輪郭（各 (N,2) のピクセル座標配列）のリストを、
    現在のパスの終端に一番近い輪郭を次々選ぶ貪欲法で並べ替える。
    輪郭は必要に応じて逆順にし、繋ぎのジャンプ距離をなるべく短くする。
    """
    remaining = list(contours)
    ordered = [remaining.pop(0)]
    current_end = ordered[0][-1]

    while remaining:
        best_i, best_dist, best_reversed = None, np.inf, False
        for i, c in enumerate(remaining):
            d_start = np.linalg.norm(current_end - c[0])
            d_end = np.linalg.norm(current_end - c[-1])
            if d_start < best_dist:
                best_dist, best_i, best_reversed = d_start, i, False
            if d_end < best_dist:
                best_dist, best_i, best_reversed = d_end, i, True

        c = remaining.pop(best_i)
        if best_reversed:
            c = c[::-1]
        ordered.append(c)
        current_end = c[-1]

    return ordered


def resample_by_arclength(z: np.ndarray, n_points: int, is_jump_segment=None):
    """
    複素数列 z（閉曲線）を弧長基準で等間隔に n_points 点へ再サンプリングする。

    is_jump_segment: z の各区間(z[i]→z[i+1], 最後は z[-1]→z[0])が
    「パーツ間の移動線」かどうかを示す真偽値配列(len(z),)。指定した場合、
    リサンプリング後の各点がその移動線区間上にあるかを示す jump_mask も返す。
    """
    # 閉曲線にするため最初の点を末尾に追加
    z_closed = np.concatenate([z, z[:1]])
    seg_len = np.abs(np.diff(z_closed))
    cum_len = np.concatenate([[0], np.cumsum(seg_len)])
    total_len = cum_len[-1]
    if total_len == 0:
        raise ValueError("輪郭の長さが0です。")

    target = np.linspace(0, total_len, n_points, endpoint=False)
    resampled = np.interp(target, cum_len, z_closed.real) + \
                1j * np.interp(target, cum_len, z_closed.imag)

    if is_jump_segment is None:
        return resampled

    # 各 target 位置がどの元の区間(セグメント)に入るかを求め、
    # そのセグメントがジャンプ区間なら True にする
    seg_idx = np.searchsorted(cum_len, target, side="right") - 1
    seg_idx = np.clip(seg_idx, 0, len(is_jump_segment) - 1)
    jump_mask = np.asarray(is_jump_segment)[seg_idx]
    return resampled, jump_mask


# ----------------------------------------------------------------------
# 2. DFT でエピサイクル（半径・角速度・初期位相）を求める
# ----------------------------------------------------------------------
def compute_epicycles(z: np.ndarray):
    """
    z_n の DFT を計算し、各周波数 k について
      半径 radius, 角速度 freq(=k, ただし負の周波数も含む), 初期位相 phase
    を返す。半径の大きい順にソートして返す（見た目の寄与が大きい円から描画するため）。
    """
    N = len(z)
    Y = np.fft.fft(z) / N          # Y(k)/N が各エピサイクルの複素振幅
    freqs = np.fft.fftfreq(N, d=1.0 / N)  # k = ..., -2, -1, 0, 1, 2, ... (整数)

    radius = np.abs(Y)
    phase = np.angle(Y)

    order = np.argsort(-radius)    # 半径が大きい順
    return freqs[order], radius[order], phase[order]


# ----------------------------------------------------------------------
# 3. アニメーション作成
# ----------------------------------------------------------------------
def make_animation(freqs, radius, phase, n_circles=None,
                    frames=300, fps=30, out_path="output.gif",
                    trace_len=None):
    N = len(freqs)
    if n_circles is None:
        n_circles = N
    n_circles = min(n_circles, N)

    freqs = freqs[:n_circles]
    radius = radius[:n_circles]
    phase = phase[:n_circles]

    fig, ax = plt.subplots(figsize=(7, 7))
    max_r = np.sum(radius) * 1.15 + 1
    ax.set_xlim(-max_r, max_r)
    ax.set_ylim(-max_r, max_r)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")

    # 円・ベクトル線を描くための Line2D たち
    circle_lines = [ax.plot([], [], color="gray", lw=0.6, alpha=0.5)[0]
                     for _ in range(n_circles)]
    vector_line, = ax.plot([], [], color="cyan", lw=1.2)
    trace_line, = ax.plot([], [], color="white", lw=1.8)

    theta = np.linspace(0, 2 * np.pi, 60)

    trace_x, trace_y = [], []
    if trace_len is None:
        trace_len = frames  # 1周期分をすべて残す（絵全体が残る）

    def epicycle_positions(t):
        """時刻 t(0〜1, 1周期で1周)における各エピサイクル中心の位置列(先頭=原点)を返す"""
        angle = 2 * np.pi * freqs * t + phase
        vecs = radius * np.exp(1j * angle)
        centers = np.concatenate([[0j], np.cumsum(vecs)])
        return centers  # centers[-1] が合成先端＝実際に絵を描く点

    def init():
        vector_line.set_data([], [])
        trace_line.set_data([], [])
        for cl in circle_lines:
            cl.set_data([], [])
        return circle_lines + [vector_line, trace_line]

    def update(frame):
        t = frame / frames
        centers = epicycle_positions(t)

        # 各円とベクトルを描画
        for i, cl in enumerate(circle_lines):
            cx, cy = centers[i].real, centers[i].imag
            r = radius[i]
            cl.set_data(cx + r * np.cos(theta), cy + r * np.sin(theta))

        vector_line.set_data(centers.real, centers.imag)

        tip = centers[-1]
        trace_x.append(tip.real)
        trace_y.append(tip.imag)
        if len(trace_x) > trace_len:
            del trace_x[0]
            del trace_y[0]
        trace_line.set_data(trace_x, trace_y)

        return circle_lines + [vector_line, trace_line]

    anim = FuncAnimation(fig, update, frames=frames, init_func=init,
                          interval=1000 / fps, blit=True)

    if out_path.lower().endswith(".gif"):
        anim.save(out_path, writer=PillowWriter(fps=fps))
    else:
        anim.save(out_path, fps=fps, dpi=150)

    plt.close(fig)
    print(f"保存しました: {out_path}")


def save_static_image(freqs, radius, phase, n_points, out_path,
                       n_circles=None, style="light", show_circles=False,
                       jump_mask=None, remove_jumps=False):
    """
    完成した絵（全エピサイクルを合成した1周期分の軌跡）を1枚の静止画として保存する。

    style:        "light" (白背景に黒線, 印刷向け) / "dark" (黒背景に白線, アニメと同じ配色)
    show_circles: True の場合、最終フレーム時点のエピサイクル円も薄く重ねて描く
    jump_mask:    image_to_complex_points が返す、各点がパーツ間の移動線
                  (輪郭→目→口 のような継ぎ目)上にあるかを示す真偽値配列
    remove_jumps: True の場合、jump_mask で示された継ぎ目の線を描画から除外する
                  （＝輪郭を描いたあとの移動線を消したクリーンな絵になる）
    """
    N = len(freqs)
    if n_circles is None:
        n_circles = N
    n_circles = min(n_circles, N)

    freqs_c = freqs[:n_circles]
    radius_c = radius[:n_circles]
    phase_c = phase[:n_circles]

    # 1周期分（=元の絵1つぶん）の軌跡を高精細に再構成
    # ※ n_points は DFT計算時と同じ点数・同じ時刻グリッドを使うことで、
    #   jump_mask の各点とここでの再構成点が1対1に対応する
    t = np.linspace(0, 1, n_points, endpoint=False)
    angle = 2 * np.pi * freqs_c[:, None] * t[None, :] + phase_c[:, None]
    vecs = radius_c[:, None] * np.exp(1j * angle)
    trace = np.sum(vecs, axis=0)
    trace = np.concatenate([trace, trace[:1]])  # 始点に戻して閉じる

    trace_x, trace_y = trace.real.copy(), trace.imag.copy()

    if remove_jumps and jump_mask is not None:
        mask_closed = np.concatenate([jump_mask, jump_mask[:1]])
        # ジャンプ区間上の点は NaN にして、その部分で線を切る(=描かない)
        trace_x[mask_closed] = np.nan
        trace_y[mask_closed] = np.nan

    if style == "dark":
        bg, line_color, circle_color = "black", "white", "gray"
    else:
        bg, line_color, circle_color = "white", "black", "lightgray"

    fig, ax = plt.subplots(figsize=(7, 7))
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)

    if show_circles:
        centers = np.concatenate([[0j], np.cumsum(vecs[:, -1])])
        theta = np.linspace(0, 2 * np.pi, 60)
        for i in range(n_circles):
            cx, cy = centers[i].real, centers[i].imag
            r = radius_c[i]
            ax.plot(cx + r * np.cos(theta), cy + r * np.sin(theta),
                    color=circle_color, lw=0.6, alpha=0.6)

    ax.plot(trace_x, trace_y, color=line_color, lw=1.8)

    max_r = max(np.nanmax(np.abs(trace.real)), np.nanmax(np.abs(trace.imag))) * 1.15 + 1
    ax.set_xlim(-max_r, max_r)
    ax.set_ylim(-max_r, max_r)
    ax.set_aspect("equal")
    ax.axis("off")

    plt.savefig(out_path, dpi=200, facecolor=bg, bbox_inches="tight")
    plt.close(fig)
    print(f"保存しました: {out_path}")


# ----------------------------------------------------------------------
# メイン処理
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="フーリエ級数(エピサイクル)グラフアート生成")
    parser.add_argument("--image", required=True, help="入力画像のパス")
    parser.add_argument("--out", default="output.gif", help="出力アニメーションファイル(.gif/.mp4)")
    parser.add_argument("--n_points", type=int, default=800, help="パスから間引くサンプル点数")
    parser.add_argument("--n_circles", type=int, default=None, help="使用するエピサイクル数(既定:全部)")
    parser.add_argument("--frames", type=int, default=None, help="1周期のフレーム数(既定=n_points)")
    parser.add_argument("--fps", type=int, default=30, help="フレームレート")
    parser.add_argument("--mode", choices=["skeleton", "edge", "silhouette"], default="skeleton",
                         help="skeleton(既定): 中心線を1本の線として辿る(綺麗) / edge: 旧方式(輪郭の縁を辿る) / silhouette: 塗りつぶし外形のみ")
    parser.add_argument("--canny1", type=int, default=50, help="Cannyエッジ検出の下側しきい値(edgeモードのみ)")
    parser.add_argument("--canny2", type=int, default=150, help="Cannyエッジ検出の上側しきい値(edgeモードのみ)")
    parser.add_argument("--min_contour_len", type=int, default=15, help="採用するパーツの最小点数(ノイズ除去)")
    parser.add_argument("--max_dim", type=int, default=600, help="処理前に画像の長辺をこのピクセル数に縮小")
    parser.add_argument("--png", default=None,
                         help="静止画(継ぎ目の線あり)の出力先(既定: --out と同名の.pngファイル)")
    parser.add_argument("--png_clean", default=None,
                         help="静止画(継ぎ目の線を消したクリーン版)の出力先(既定: <png>_clean.png)")
    parser.add_argument("--no_png", action="store_true", help="継ぎ目の線ありの静止画を出力しない")
    parser.add_argument("--no_png_clean", action="store_true", help="継ぎ目の線を消したクリーン版を出力しない")
    parser.add_argument("--static_style", choices=["light", "dark"], default="light",
                         help="静止画の配色: light=白背景黒線(既定) / dark=アニメと同じ黒背景白線")
    parser.add_argument("--static_show_circles", action="store_true",
                         help="静止画にも最終時点のエピサイクル円を薄く重ねて描く")
    args = parser.parse_args()

    frames = args.frames or args.n_points

    print("画像から輪郭を抽出中...")
    z, jump_mask = image_to_complex_points(
        args.image, n_points=args.n_points,
        mode=args.mode, canny1=args.canny1,
        canny2=args.canny2,
        min_contour_len=args.min_contour_len,
        max_dim=args.max_dim)

    print("DFTでエピサイクルを計算中...")
    freqs, radius, phase = compute_epicycles(z)

    print("アニメーションを生成中...")
    make_animation(freqs, radius, phase,
                   n_circles=args.n_circles,
                   frames=frames, fps=args.fps,
                   out_path=args.out)

    base = args.out.rsplit(".", 1)[0]

    if not args.no_png:
        png_path = args.png or (base + ".png")
        print("静止画(継ぎ目の線あり)を生成中...")
        save_static_image(freqs, radius, phase, n_points=args.n_points,
                           out_path=png_path, n_circles=args.n_circles,
                           style=args.static_style,
                           show_circles=args.static_show_circles,
                           jump_mask=jump_mask, remove_jumps=False)

    if not args.no_png_clean:
        png_clean_path = args.png_clean or (base + "_clean.png")
        print("静止画(継ぎ目の線を消したクリーン版)を生成中...")
        save_static_image(freqs, radius, phase, n_points=args.n_points,
                           out_path=png_clean_path, n_circles=args.n_circles,
                           style=args.static_style,
                           show_circles=args.static_show_circles,
                           jump_mask=jump_mask, remove_jumps=True)


if __name__ == "__main__":
    main()
