using System.Drawing;
using System.Drawing.Imaging;
using NumSharp;
using SixLabors.ImageSharp.Formats.Bmp;

public static class BitmapExtensions
{
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