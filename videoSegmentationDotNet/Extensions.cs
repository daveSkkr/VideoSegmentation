using System.Drawing;
using System.Drawing.Imaging;
using SixLabors.ImageSharp.Formats.Bmp;

public static class Extensions
{

    /// <summary>
    /// Converts the image data into a byte array.
    /// </summary>
    /// <param name="imageIn">The image to convert to an array</param>
    /// <param name="fmt">The format to save the image in</param>
    /// <returns>An array of bytes</returns>
    public static byte[] ToArray(this global::System.Drawing.Bitmap imageIn, ImageFormat fmt)
    {
        using (MemoryStream ms = new MemoryStream())
        {
            imageIn.Save(ms, fmt);
            return ms.ToArray();
        }
    }

    public static Bitmap ToBitmap(this global::SixLabors.ImageSharp.Image<Rgba32> imageIn)
    {
        using (MemoryStream ms = new MemoryStream())
        {
            imageIn.Save(ms, new BmpEncoder());
            return new Bitmap(ms);
        }
    }
}