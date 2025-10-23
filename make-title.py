from PIL import Image, ImageDraw, ImageFont


def make_chapter_title(
    title,
    filepath,
    font_path,
    width=800,
    height=200,
):
    # Image dimensions and settings
    background_color = "white"
    text_color = "black"
    font_size = 40

    # Create a blank image with white background
    image = Image.new("RGB", (width, height), background_color)
    draw = ImageDraw.Draw(image)

    try:
        # Load font
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        raise RuntimeError(
            f"Font file not found at {font_path}. Please provide a valid font path."
        )

    # Calculate text size and position
    text_bbox = draw.textbbox((0, 0), title, font=font)  # Get bounding box of the text
    text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
    text_x = (width - text_width) // 2
    text_y = (height - text_height) // 2

    # Draw the text onto the image
    draw.text((text_x, text_y), title, fill=text_color, font=font)

    # Save the image to the specified filepath
    image.save(filepath, "PNG")


# example usage
if __name__ == "__main__":
    make_chapter_title(
        "hey you", "test-2TC-R.png", "fonts/GenSenRounded2TC-R.otf", 800, 200
    )

    print("Done")
