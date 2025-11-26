from PIL import Image, ImageDraw
import math

def create_test_image():
    img = Image.new('RGB', (400, 400), color=(240, 240, 240))
    draw = ImageDraw.Draw(img)

    center = (200, 200)
    radius = 150

    draw.ellipse([center[0]-radius, center[1]-radius, center[0]+radius, center[1]+radius],
                 outline=(100, 100, 100), width=2)

    num_branches = 8
    for i in range(num_branches):
        angle = (2 * math.pi * i) / num_branches
        x = int(center[0] + radius * math.cos(angle))
        y = int(center[1] + radius * math.sin(angle))
        draw.line([center, (x, y)], fill=(50, 50, 200), width=2)

        sub_radius = radius // 2
        for j in range(4):
            sub_angle = angle + (math.pi / 8) * (j - 1.5)
            sx = int(center[0] + sub_radius * math.cos(sub_angle))
            sy = int(center[1] + sub_radius * math.sin(sub_angle))
            draw.line([(x, y), (sx, sy)], fill=(200, 50, 50), width=1)

    for i in range(3):
        r = 50 + i * 40
        draw.ellipse([center[0]-r, center[1]-r, center[0]+r, center[1]+r],
                     outline=(150, 150, 150), width=1)

    img.save('test_image.jpg')
    print("Test image created: test_image.jpg")

if __name__ == "__main__":
    create_test_image()
