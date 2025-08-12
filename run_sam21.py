import cv2
import numpy as np
import requests
from ultralytics import SAM
import matplotlib.pyplot as plt


def interactive_sam_segmentation(image_url="https://ultralytics.com/images/bus.jpg"):
    """
    마우스로 바운딩 박스를 그려서 SAM 2.1 모델로 객체 분할을 수행하는 함수
    """
    # 전역 변수
    drawing = False
    ix, iy = -1, -1
    bbox = []

    def draw_rectangle(event, x, y, flags, param):
        """마우스 콜백 함수 - 바운딩 박스 그리기"""
        nonlocal ix, iy, drawing, img, bbox

        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            ix, iy = x, y
            bbox = []  # 새로운 박스를 그릴 때 이전 박스 초기화

        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing:
                img_copy = img.copy()
                cv2.rectangle(img_copy, (ix, iy), (x, y), (0, 255, 0), 2)
                cv2.imshow("SAM 2.1 - Draw Bounding Box", img_copy)

        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            cv2.rectangle(img, (ix, iy), (x, y), (0, 255, 0), 2)
            bbox = [ix, iy, x, y]
            cv2.imshow("SAM 2.1 - Draw Bounding Box", img)
            print(f"바운딩 박스 좌표: {bbox}")

    # 이미지 다운로드
    print("이미지를 다운로드하는 중...")
    response = requests.get(image_url, stream=True)
    response.raise_for_status()
    image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
    img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    img_original = img.copy()  # SAM 모델용 원본 이미지 보존

    # 윈도우 설정
    cv2.namedWindow("SAM 2.1 - Draw Bounding Box", cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback("SAM 2.1 - Draw Bounding Box", draw_rectangle)

    print("=" * 60)
    print("🎯 SAM 2.1 Interactive Segmentation")
    print("=" * 60)
    print("📖 사용법:")
    print(
        "  1. 마우스로 클릭&드래그하여 분할하고 싶은 객체 주위에 바운딩 박스를 그리세요"
    )
    print("  2. 바운딩 박스를 그린 후 아무 키나 누르면 SAM 2.1 모델이 실행됩니다")
    print("  3. 'q' 키를 누르면 종료됩니다")
    print("  4. 'r' 키를 누르면 이미지를 리셋합니다")
    print("=" * 60)

    # 메인 루프
    while True:
        cv2.imshow("SAM 2.1 - Draw Bounding Box", img)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):  # 'q' 키로 종료
            print("프로그램을 종료합니다.")
            break

        elif key == ord("r"):  # 'r' 키로 리셋
            img = img_original.copy()
            bbox = []
            print("이미지가 리셋되었습니다. 새로운 바운딩 박스를 그려주세요.")

        elif key != 255 and bbox:  # 다른 키를 누르고 bbox가 있으면 SAM 실행
            print("\n🔄 SAM 2.1 모델을 실행하는 중...")
            try:
                # SAM 2.1 모델 로드
                model = SAM("sam2.1_b.pt")

                # 바운딩 박스로 추론 실행
                results = model(img_original, bboxes=[bbox])

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

                else:
                    print("❌ 분할된 객체가 없습니다. 다른 영역을 시도해보세요.")

            except Exception as e:
                print(f"❌ 오류 발생: {str(e)}")

            # 새로운 바운딩 박스를 위해 리셋
            img = img_original.copy()
            bbox = []
            print("\n새로운 바운딩 박스를 그리거나 'q'를 눌러 종료하세요.")

    cv2.destroyAllWindows()
    print("프로그램이 종료되었습니다.")


# 함수 실행
interactive_sam_segmentation()
