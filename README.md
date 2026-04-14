# Camera_pose_Estimation_and_AR
이 프로젝트는 Open CV를 활용하여 카메라의 파라미터를 추출하고, 체스판 패턴을 인식하여그 위에 3D 사각뿔(Pyramid)을 증강현실로 구현하는 프로그램입니다.

## 주요 기능
1. Camera Calibration: 비디오 프레임에서 체스판 코너를 검출하여 카메라 렬과 왜곡 계수를 산출합니다.
2. Pose Estimation: solvePnP 알고리즘을 사용하여 카메라의 위치와 회전 정보를 실시간으로 추적합니다.
3. AR Visualization: 체스판의 특정 그리드 좌표를 기준으로 밑면 1X1, 높이 1 규격의 사각뿔을 랜더링합니다.
4. Undistortion: 캘리브레이션 데이터를 바탕으로 영상의 렌즈왜곡을 실시간으로 보정하는 기능을 포함합니다.

## 실행 환경
1. Languege: Python 3.x
2. Libraries: numpy, opencv-python, json, os

## 사용 방법
1. 필요 데이터: chessboard.mp4(카메라 캘리브레이션 및 AR 테스트를 위한 10X7 크기의 체스판 주행 영상)
2. 캘리브레이션 단계: 최초 실행 시 calib_result.json 파일이 없다면 자동으로 이미지 선택 모드가 시작됩니다.
   * space: 현재 프레임 정지 및 체스판 패턴 인식 확인
   * Enter: 해당 프레임을 캘리브레이션 데이터로 저장
   * ESC: 이미지 선택 완료 및 데이터 계산 시작
3. AR 모드: 계산된 데이터를 불러온 후 실시간으로 사각뿔이 표시됩니다.
   * Tab: 왜곡 보정 모드 On/Off 전환
   * Space: 영상 일시정지
   * ESC: 프로그램 종료

![실행 화면 캡처](./screenshot001.png)
