use image::{open, GenericImageView, ImageBuffer, Rgba};
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Paths for input image, mask, and output
    let input_path = "path/to/input.jpg";
    let mask_path = "path/to/mask.png";
    let output_path = "path/to/output.png";

    // Open the input image and mask
    let input_image = open(input_path)?;
    let mask_image = open(mask_path)?;

    // Ensure the dimensions match
    assert_eq!(input_image.dimensions(), mask_image.dimensions());

    let (width, height) = input_image.dimensions();

    // Create a new image with an alpha channel
    let mut output_image = ImageBuffer::new(width, height);

    for (x, y, pixel) in input_image.pixels() {
        let mask_pixel = mask_image.get_pixel(x, y);
        
        // Check if the mask pixel is white (255, 255, 255)
        let alpha = if mask_pixel[0] > 128 { 255 } else { 0 };

        let rgba = Rgba([pixel[0], pixel[1], pixel[2], alpha]);
        output_image.put_pixel(x, y, rgba);
    }

    // Save the output image
    output_image.save(Path::new(output_path))?;

    println!("Converted image saved to {}", output_path);

    Ok(())
}
