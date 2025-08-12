import cv2
import numpy as np
import requests
from ultralytics import SAM
import matplotlib.pyplot as plt


def interactive_sam_point_segmentation(
    image_url="https://ultralytics.com/images/bus.jpg",
):
    """
    마우스 클릭으로 포인트를 입력하여 SAM 2.1 모델로 객체 분할을 수행하는 함수
    """
    # 전역 변수
    points = []  # 클릭한 포인트들을 저장

    def mouse_callback(event, x, y, flags, param):
        """마우스 콜백 함수 - 포인트 수집"""
        nonlocal points, img_display

        if event == cv2.EVENT_LBUTTONDOWN:
            # 왼쪽 클릭 시 포인트 추가
            points.append([x, y])

            # 이미지에 포인트 표시
            img_display = img_original.copy()
            for i, point in enumerate(points):
                # 포인트를 원으로 표시 (빨간색)
                cv2.circle(img_display, tuple(point), 5, (0, 0, 255), -1)
                # 포인트 번호 표시
                cv2.putText(
                    img_display,
                    str(i + 1),
                    (point[0] + 10, point[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )

            cv2.imshow("SAM 2.1 - Click Points", img_display)
            print(f"포인트 {len(points)} 추가됨: [{x}, {y}]")
            print(f"현재 포인트들: {points}")

        elif event == cv2.EVENT_RBUTTONDOWN:
            # 오른쪽 클릭 시 마지막 포인트 제거
            if points:
                removed_point = points.pop()
                print(f"포인트 제거됨: {removed_point}")

                # 이미지 업데이트
                img_display = img_original.copy()
                for i, point in enumerate(points):
                    cv2.circle(img_display, tuple(point), 5, (0, 0, 255), -1)
                    cv2.putText(
                        img_display,
                        str(i + 1),
                        (point[0] + 10, point[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2,
                    )

                cv2.imshow("SAM 2.1 - Click Points", img_display)
                print(f"현재 포인트들: {points}")

    # 이미지 다운로드
    print("이미지를 다운로드하는 중...")
    response = requests.get(image_url, stream=True)
    response.raise_for_status()
    image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
    img_original = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    img_display = img_original.copy()

    # 윈도우 설정
    cv2.namedWindow("SAM 2.1 - Click Points", cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback("SAM 2.1 - Click Points", mouse_callback)

    print("=" * 70)
    print("🎯 SAM 2.1 Interactive Point Segmentation")
    print("=" * 70)
    print("📖 사용법:")
    print("  1. 왼쪽 마우스 클릭: 분할하고 싶은 객체 위에 포인트 추가")
    print("  2. 오른쪽 마우스 클릭: 마지막 포인트 제거")
    print("  3. SPACE 키: SAM 2.1 모델 실행 (포인트가 1개 이상 있어야 함)")
    print("  4. 'c' 키: 모든 포인트 클리어")
    print("  5. 'r' 키: 이미지 리셋")
    print("  6. 'q' 키: 프로그램 종료")
    print("=" * 70)
    print("💡 팁: 여러 포인트를 추가하면 더 정확한 분할이 가능합니다!")
    print("=" * 70)

    # 메인 루프
    while True:
        cv2.imshow("SAM 2.1 - Click Points", img_display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):  # 'q' 키로 종료
            print("프로그램을 종료합니다.")
            break

        elif key == ord("c"):  # 'c' 키로 포인트 클리어
            points = []
            img_display = img_original.copy()
            cv2.imshow("SAM 2.1 - Click Points", img_display)
            print("모든 포인트가 클리어되었습니다.")

        elif key == ord("r"):  # 'r' 키로 리셋
            points = []
            img_display = img_original.copy()
            cv2.imshow("SAM 2.1 - Click Points", img_display)
            print("이미지가 리셋되었습니다. 새로운 포인트를 클릭해주세요.")

        elif key == ord(" "):  # SPACE 키로 SAM 실행
            if len(points) == 0:
                print("❌ 포인트를 최소 1개 이상 클릭해주세요!")
                continue

            print(f"\n🔄 SAM 2.1 모델을 실행하는 중... (포인트 개수: {len(points)})")
            print(f"입력 포인트: {points}")

            try:
                # SAM 2.1 모델 로드
                model = SAM("sam2.1_b.pt")

                # 포인트로 추론 실행
                if len(points) == 1:
                    # 단일 포인트
                    results = model(img_original, points=points[0])
                else:
                    # 다중 포인트
                    results = model(img_original, points=points)

                if results and len(results) > 0:
                    print("✅ 분할 완료! 결과를 표시합니다.")
                    results[0].show()

                    # 추가 정보 출력
                    if hasattr(results[0], "masks") and results[0].masks is not None:
                        num_masks = len(results[0].masks)
                        print(f"📊 분할된 마스크 수: {num_masks}")

                        # 각 마스크의 신뢰도 점수 출력 (있는 경우)
                        if (
                            hasattr(results[0], "boxes")
                            and results[0].boxes is not None
                        ):
                            if hasattr(results[0].boxes, "conf"):
                                confs = results[0].boxes.conf
                                for i, conf in enumerate(confs):
                                    print(f"  마스크 {i+1}: 신뢰도 {conf:.3f}")

                    print(f"\n사용된 포인트 개수: {len(points)}")
                    for i, point in enumerate(points):
                        print(f"  포인트 {i+1}: {point}")

                else:
                    print(
                        "❌ 분할된 객체가 없습니다. 다른 위치에 포인트를 시도해보세요."
                    )

            except Exception as e:
                print(f"❌ 오류 발생: {str(e)}")

            print("\n계속하려면 새로운 포인트를 추가하거나 다른 키를 눌러주세요.")

    cv2.destroyAllWindows()
    print("프로그램이 종료되었습니다.")


# 함수 실행
interactive_sam_point_segmentation()
