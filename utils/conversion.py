def convert_pixel_to_meters(pixel_distance, refrence_height_in_meters, refrence_height_in_pixels):
    return (pixel_distance * refrence_height_in_meters) / refrence_height_in_pixels

def convert_meters_to_pixel(meters, refrence_height_in_meters, refrence_height_in_pixels):
    return (meters * refrence_height_in_pixels) / refrence_height_in_meters