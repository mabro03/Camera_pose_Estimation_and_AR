import numpy as np
import cv2 as cv
import json

def load_calib(file_path):
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
        return np.array(data['K']), np.array(data['dist']), data.get('width', 800)
    except FileNotFoundError:
        print("Error: 'calib_result.json' 파일이 없습니다. 먼저 캘리브레이션을 수행하세요.")
        return None, None, None

if __name__ == '__main__':
    VIDEO_PATH = 'chessboard.mp4'
    K, dist_coeff, TW = load_calib("calib_result.json")

    if K is not None:
        video = cv.VideoCapture(VIDEO_PATH)
        show_rectify = True
        map1, map2 = None, None

        print("[Tab]: 보정 On/Off 전환 | [ESC]: 종료")

        while True:
            valid, img = video.read()
            if not valid: break

            # --- 이미지 리사이징 (Calibration 때와 동일한 크기) ---
            aspect_ratio = img.shape[0] / img.shape[1]
            target_height = int(TW * aspect_ratio)
            img = cv.resize(img, (TW, target_height), interpolation=cv.INTER_AREA)

            display = img.copy()
            if show_rectify:
                if map1 is None:
                    h, w = img.shape[:2]
                    # 왜곡 보정을 위한 매핑 계산
                    map1, map2 = cv.initUndistortRectifyMap(K, dist_coeff, None, None, (w, h), cv.CV_32FC1)
                display = cv.remap(img, map1, map2, cv.INTER_LINEAR)

            status = "Rectified" if show_rectify else "Original"
            cv.putText(display, status, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv.imshow("Geometric Distortion Correction", display)

            key = cv.waitKey(10)
            if key == 27: break
            elif key == ord('\t'): show_rectify = not show_rectify

        video.release()
        cv.destroyAllWindows()