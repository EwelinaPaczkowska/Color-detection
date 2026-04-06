import argparse
import math
import sys
import cv2
import numpy as np

def buduj_maske_czerwieni(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    dolny_1 = np.array([0, 100, 80], dtype=np.uint8)
    gorny_1 = np.array([10, 255, 255], dtype=np.uint8)

    dolny_2 = np.array([170, 100, 80], dtype=np.uint8)
    gorny_2 = np.array([180, 255, 255], dtype=np.uint8)

    maska_1 = cv2.inRange(hsv, dolny_1, gorny_1)
    maska_2 = cv2.inRange(hsv, dolny_2, gorny_2)

    maska = cv2.bitwise_or(maska_1, maska_2)

    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))

    maska = cv2.morphologyEx(maska, cv2.MORPH_OPEN, kernel_open)
    maska = cv2.morphologyEx(maska, cv2.MORPH_CLOSE, kernel_close)

    return maska


def obiekt_z_momentow(maska_0_255, min_pole=300.0):
    maska_bin = (maska_0_255 > 0).astype(np.uint8)
    pole = float(np.count_nonzero(maska_bin))

    if pole < min_pole:
        return None, None, None

    m = cv2.moments(maska_bin, binaryImage=True)

    if abs(m["m00"]) < 1e-6:
        return None, None, None

    cx = int(m["m10"] / m["m00"])
    cy = int(m["m01"] / m["m00"])

    promien = int(round(math.sqrt(pole / math.pi)))

    return (cx, cy), pole, promien


def rysuj_paski_odchylenia(img, cx, szerokosc, y=40, wysokosc_paska=16, margines=10):
    x0 = margines
    x1 = szerokosc - margines
    x_srodek = szerokosc // 2

    cx = max(0, min(cx, szerokosc - 1))

    cv2.rectangle(img, (x0, y), (x1, y + wysokosc_paska), (255, 255, 255), 1)

    cv2.line(img, (x_srodek, y - 4), (x_srodek, y + wysokosc_paska + 4), (255, 255, 255), 1)

    if cx < x_srodek:
        cv2.rectangle(img, (cx, y + 2), (x_srodek, y + wysokosc_paska - 2), (255, 0, 0), -1)
    elif cx > x_srodek:
        cv2.rectangle(img, (x_srodek, y + 2), (cx, y + wysokosc_paska - 2), (0, 255, 0), -1)

    odchylenie = cx - x_srodek

    cv2.putText(
        img,
        f"deviation_px = {odchylenie:+d}",
        (x0, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--min-pole", type=float, default=300.0)
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)

    if not cap.isOpened():
        print(f"Nie można otworzyć pliku: {args.video}", file=sys.stderr)
        return 1

    fps = cap.get(cv2.CAP_PROP_FPS)
    opoznienie_ms = int(1000 / fps) if fps and fps > 1e-3 else 20

    while True:
        ok, klatka = cap.read()
        if not ok:
            break

        h, w = klatka.shape[:2]
        x_srodek = w // 2

        maska = buduj_maske_czerwieni(klatka)

        centrum, pole, promien = obiekt_z_momentow(maska, min_pole=args.min_pole)

        cv2.namedWindow("Obraz oryginalny", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Obraz oryginalny", 700, 400)

        cv2.namedWindow("Obraz przetworzony", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Obraz przetworzony", 700, 400)

        cv2.imshow("Obraz przetworzony", maska)

        wiz = klatka.copy()

        cv2.line(wiz, (x_srodek, 0), (x_srodek, h), (255, 255, 255), 1)

        if centrum is not None:
            cx, cy = centrum

            cv2.circle(wiz, (cx, cy), max(promien, 1), (0, 0, 255), 2)
            cv2.circle(wiz, (cx, cy), 3, (0, 0, 255), -1)

            rysuj_paski_odchylenia(wiz, cx=cx, szerokosc=w)

            cv2.putText(
                wiz,
                f"center=({cx}, {cy}), area={int(pole)}",
                (10, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        cv2.imshow("Obraz oryginalny", wiz)

        klawisz = cv2.waitKey(opoznienie_ms) & 0xFF
        if klawisz in (ord("q"), 27):
            break

    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())