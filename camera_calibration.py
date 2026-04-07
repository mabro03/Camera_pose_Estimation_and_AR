import numpy as np
import cv2 as cv
import json

def select_img_from_video(video_file, board_pattern, TW=800):
    video = cv.VideoCapture(video_file)
    if not video.isOpened():
        print(f"Error: 파일을 찾을 수 없습니다. ({video_file})")
        return []

    img_select = []
    wnd_name = 'Select Calibration Images'
    print("사용법: [Space] - 정지/확인 | [Enter] - 이미지 선택 | [ESC] - 선택 완료")

    while True:
        valid, img = video.read()
        if not valid:
            break

        # --- 이미지 리사이징 로직 추가 ---
        aspect_ratio = img.shape[0] / img.shape[1]
        target_height = int(TW * aspect_ratio)
        img = cv.resize(img, (TW, target_height), interpolation=cv.INTER_AREA)

        display = img.copy()
        cv.putText(display, f'Selected: {len(img_select)}', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv.imshow(wnd_name, display)

        key = cv.waitKey(10)
        if key == ord(' '):  # Space: 일시정지 및 패턴 확인
            complete, pts = cv.findChessboardCorners(img, board_pattern)
            cv.drawChessboardCorners(display, board_pattern, pts, complete)
            cv.imshow(wnd_name, display)
            
            sub_key = cv.waitKey()
            if sub_key == ord('\r'):  # Enter: 선택 확정
                img_select.append(img)
                print(f"이미지 {len(img_select)}장 선택됨")
        elif key == 27:  # ESC: 종료
            break

    video.release()
    cv.destroyAllWindows()
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

    if len(img_points) == 0:
        return None, None, None

    obj_pts = [[c, r, 0] for r in range(board_pattern[1]) for c in range(board_pattern[0])]
    obj_points = [np.array(obj_pts, dtype=np.float32) * board_cellsize] * len(img_points)

    rms, K, dist, rvecs, tvecs = cv.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)
    return rms, K, dist

if __name__ == '__main__':
    # 설정값 (본인의 환경에 맞춰 수정)
    VIDEO_PATH = 'chessboard.mp4'  # 파일명 확인!
    PATTERN = (10, 7)
    CELL_SIZE = 0.025 
    TARGET_WIDTH = 800  # 리사이징 가로 크기

    selected = select_img_from_video(VIDEO_PATH, PATTERN, TW=TARGET_WIDTH)
    
    if len(selected) > 0:
        rms, K, dist = run_calibration(selected, PATTERN, CELL_SIZE)
        
        if rms is not None:
            print('\n## Calibration Results')
            print(f'RMS Error: {rms}')
            print(f'Camera Matrix (K):\n{K}')
            print(f'Distortion Coeffs: {dist.flatten()}')

            # 결과 저장
            result_data = {
                "rms": rms,
                "K": K.tolist(),
                "dist": dist.tolist(),
                "width": TARGET_WIDTH # 나중을 위해 크기 저장
            }
            with open("calib_result.json", "w") as f:
                json.dump(result_data, f, indent=4)
            print("\n결과가 'calib_result.json'에 저장되었습니다.")
    else:
        print("선택된 이미지가 없습니다.")