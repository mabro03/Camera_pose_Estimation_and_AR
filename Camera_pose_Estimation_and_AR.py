import numpy as np
import cv2 as cv
import json
import os

def select_img_from_video(video_file, board_pattern, TW=800):
    video = cv.VideoCapture(video_file)
    if not video.isOpened():
        return []

    img_select = []
    wnd_name = 'Select Calibration Images'

    while True:
        valid, img = video.read()
        if not valid: break

        aspect_ratio = img.shape[0] / img.shape[1]
        target_height = int(TW * aspect_ratio)
        img = cv.resize(img, (TW, target_height), interpolation=cv.INTER_AREA)

        display = img.copy()
        cv.imshow(wnd_name, display)

        key = cv.waitKey(10)
        if key == ord(' '):
            complete, pts = cv.findChessboardCorners(img, board_pattern)
            cv.drawChessboardCorners(display, board_pattern, pts, complete)
            cv.imshow(wnd_name, display)
            
            sub_key = cv.waitKey()
            if sub_key == ord('\r'):
                img_select.append(img)
        elif key == 27:
            break

    video.release()
    cv.destroyWindow(wnd_name)
    return img_select

def run_calibration(images, board_pattern, board_cellsize):
    img_points = []
    gray = None
    for img in images:
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        complete, pts = cv.findChessboardCorners(gray, board_pattern)
        if complete:
            criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            pts2 = cv.cornerSubPix(gray, pts, (11, 11), (-1, -1), criteria)
            img_points.append(pts2)

    if len(img_points) == 0: return None, None, None

    obj_pts = [[c, r, 0] for r in range(board_pattern[1]) for c in range(board_pattern[0])]
    obj_points = [np.array(obj_pts, dtype=np.float32) * board_cellsize] * len(img_points)

    rms, K, dist, rvecs, tvecs = cv.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)
    return rms, K, dist

def save_calib(rms, K, dist, width, file_path="calib_result.json"):
    result_data = {"rms": rms, "K": K.tolist(), "dist": dist.tolist(), "width": width}
    with open(file_path, "w") as f:
        json.dump(result_data, f, indent=4)

def load_calib(file_path):
    if not os.path.exists(file_path): return None, None, None
    with open(file_path, "r") as f:
        data = json.load(f)
    return np.array(data['K']), np.array(data['dist']), data.get('width', 800)

if __name__ == '__main__':
    VIDEO_PATH = 'chessboard.mp4'
    PATTERN = (10, 7)
    CELL_SIZE = 0.025 
    TARGET_WIDTH = 800
    CALIB_FILE = "calib_result.json"

    K, dist_coeff, TW = load_calib(CALIB_FILE)

    if K is None:
        selected = select_img_from_video(VIDEO_PATH, PATTERN, TW=TARGET_WIDTH)
        if len(selected) >= 5:
            rms, K, dist_coeff = run_calibration(selected, PATTERN, CELL_SIZE)
            if rms:
                save_calib(rms, K, dist_coeff, TARGET_WIDTH, CALIB_FILE)
                TW = TARGET_WIDTH
        else:
            exit()

    video = cv.VideoCapture(VIDEO_PATH)
    show_rectify = True
    map1, map2 = None, None
    
    base_center = (4, 3) 
    pyr_base = CELL_SIZE * np.array([
        [base_center[0], base_center[1], 0],
        [base_center[0]+1, base_center[1], 0],
        [base_center[0]+1, base_center[1]+1, 0],
        [base_center[0], base_center[1]+1, 0]
    ], dtype=np.float32)
    
    pyr_tip = CELL_SIZE * np.array([
        [base_center[0]+0.5, base_center[1]+0.5, -1]
    ], dtype=np.float32)

    obj_points = CELL_SIZE * np.array([[c, r, 0] for r in range(PATTERN[1]) for c in range(PATTERN[0])])
    board_criteria = cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_NORMALIZE_IMAGE + cv.CALIB_CB_FAST_CHECK

    while True:
        valid, img = video.read()
        if not valid: break

        aspect_ratio = img.shape[0] / img.shape[1]
        img = cv.resize(img, (TW, int(TW * aspect_ratio)), interpolation=cv.INTER_AREA)

        if show_rectify:
            if map1 is None:
                h, w = img.shape[:2]
                map1, map2 = cv.initUndistortRectifyMap(K, dist_coeff, None, None, (w, h), cv.CV_32FC1)
            display = cv.remap(img, map1, map2, cv.INTER_LINEAR)
        else:
            display = img.copy()

        success, img_points = cv.findChessboardCorners(img, PATTERN, board_criteria)
        if success:
            ret, rvec, tvec = cv.solvePnP(obj_points, img_points, K, dist_coeff)
            curr_dist = np.zeros(5) if show_rectify else dist_coeff
            
            pts_base_2d, _ = cv.projectPoints(pyr_base, rvec, tvec, K, curr_dist)
            pts_tip_2d, _ = cv.projectPoints(pyr_tip, rvec, tvec, K, curr_dist)
            
            pts_base = np.int32(pts_base_2d).reshape(-1, 2)
            pt_tip = tuple(np.int32(pts_tip_2d).get().flatten()) if hasattr(pts_tip_2d, 'get') else tuple(np.int32(pts_tip_2d).flatten())

            cv.polylines(display, [pts_base], True, (255, 0, 0), 2)
            
            for i in range(4):
                cv.line(display, tuple(pts_base[i]), pt_tip, (0, 255, 0), 2)

        cv.imshow("Integrated System", display)

        key = cv.waitKey(10)
        if key == 27: break
        elif key == ord('\t'): 
            show_rectify = not show_rectify
        elif key == ord(' '): 
            cv.waitKey()

    video.release()
    cv.destroyAllWindows()