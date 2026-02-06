"""
Simple test script for manual control of the 2-wheeler robot.
This demonstrates the API without requiring a GUI viewer.
"""
import numpy as np
from simple_wheeler_env import SimpleWheelerEnv


def print_camera_ascii(img, width=32):
    """Print a simple ASCII representation of the camera image."""
    # Convert to grayscale
    gray = np.mean(img, axis=2).astype(np.uint8)

    # Resize to target width while maintaining aspect ratio
    h, w = gray.shape
    new_h = int(h * width / w)

    # Simple nearest neighbor resize
    gray_resized = np.zeros((new_h, width), dtype=np.uint8)
    for i in range(new_h):
        for j in range(width):
            orig_i = int(i * h / new_h)
            orig_j = int(j * w / width)
            gray_resized[i, j] = gray[orig_i, orig_j]

    # ASCII characters from dark to bright
    chars = " .:-=+*#%@"

    ascii_img = []
    for row in gray_resized:
        ascii_row = ""
        for pixel in row:
            idx = int(pixel * (len(chars) - 1) / 255)
            ascii_row += chars[idx]
        ascii_img.append(ascii_row)

    return "\n".join(ascii_img)


def test_basic_api():
    """Test basic environment API."""
    print("=" * 60)
    print("Simple Wheeler Environment - Basic API Test")
    print("=" * 60)
    print()

    # Create environment with 16x16 camera
    print("Creating environment with 16x16 camera...")
    env = SimpleWheelerEnv(render_width=16, render_height=16)
    print("✓ Environment created")
    print()

    # Get initial state
    print("Initial State:")
    print(f"  Bot position: {env.get_bot_position()}")
    print(f"  Target position: {env.get_target_position()}")
    print(f"  Distance to target: {env.get_distance_to_target():.2f} units")

    camera_img = env.get_camera_image()
    print(f"  Camera shape: {camera_img.shape}")
    print(f"  Camera dtype: {camera_img.dtype}")
    print(f"  Camera value range: [{camera_img.min()}, {camera_img.max()}]")
    print()

    # Test motor controls
    print("Testing Motor Controls:")
    print("-" * 60)

    test_cases = [
        ("No movement", 0.0, 0.0, 50),
        ("Forward", 0.5, 0.5, 100),
        ("Turn right", 0.5, -0.5, 100),
        ("Turn left", -0.5, 0.5, 100),
        ("Backward", -0.5, -0.5, 50),
        ("Forward again", 0.5, 0.5, 200),
    ]

    for description, left, right, steps in test_cases:
        print(f"\n{description} (L={left:+.1f}, R={right:+.1f}) for {steps} steps:")

        start_pos = env.get_bot_position().copy()
        start_dist = env.get_distance_to_target()

        for _ in range(steps):
            camera_img = env.step(left, right)

        end_pos = env.get_bot_position()
        end_dist = env.get_distance_to_target()

        movement = np.linalg.norm(end_pos - start_pos)
        dist_change = end_dist - start_dist

        print(f"  Position: {end_pos[:2]} (moved {movement:.4f} units)")
        print(f"  Distance to target: {end_dist:.2f} (Δ={dist_change:+.2f})")

    print()
    print("-" * 60)
    print("Camera View (ASCII representation):")
    print("-" * 60)
    camera_img = env.get_camera_image()
    print(print_camera_ascii(camera_img, width=16))
    print("-" * 60)

    # Check if orange is visible
    # Orange is approximately [255, 127, 0] in RGB
    # Let's check for pixels with high red, medium green, low blue
    is_orange = (camera_img[:,:,0] > 200) & (camera_img[:,:,1] > 50) & (camera_img[:,:,1] < 150) & (camera_img[:,:,2] < 100)
    orange_pixels = np.sum(is_orange)
    print(f"\nOrange-ish pixels detected: {orange_pixels} / {camera_img.shape[0] * camera_img.shape[1]}")

    if orange_pixels > 0:
        print("✓ Target cube may be visible in camera!")
    else:
        print("✗ Target cube not visible (bot may need to turn or move)")

    print()
    print("Final State:")
    print(f"  Bot position: {env.get_bot_position()}")
    print(f"  Distance to target: {env.get_distance_to_target():.2f} units")
    print()

    env.close()
    print("✓ Test complete!")


def test_driving_towards_target():
    """Test driving straight and monitoring distance to target."""
    print("\n" + "=" * 60)
    print("Test: Driving Forward Towards Target")
    print("=" * 60)
    print()

    env = SimpleWheelerEnv(render_width=16, render_height=16)

    print("Driving forward with both motors at 0.5 for 500 steps...")
    print()

    initial_dist = env.get_distance_to_target()
    print(f"Initial distance: {initial_dist:.2f}")

    # Drive forward
    for i in range(500):
        camera_img = env.step(0.5, 0.5)

        if i % 100 == 0:
            dist = env.get_distance_to_target()
            pos = env.get_bot_position()
            print(f"  Step {i:3d}: pos=({pos[0]:7.4f}, {pos[1]:7.4f}), dist={dist:.2f}")

    final_dist = env.get_distance_to_target()
    print(f"\nFinal distance: {final_dist:.2f}")
    print(f"Distance change: {final_dist - initial_dist:+.2f}")

    if final_dist < initial_dist:
        print("✓ Bot moved closer to target!")
    else:
        print("✗ Bot moved away from target or stayed same")

    env.close()
    print()


if __name__ == "__main__":
    test_basic_api()
    test_driving_towards_target()

    print("\n" + "=" * 60)
    print("All tests complete!")
    print()
    print("Next steps:")
    print("  1. Use SimpleWheelerEnv class in your own scripts")
    print("  2. Experiment with different motor commands")
    print("  3. Process camera images for navigation")
    print("  4. Implement reward function based on distance")
    print("  5. Train a neural network to control the bot")
    print("=" * 60)
