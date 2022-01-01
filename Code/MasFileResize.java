import org.apache.commons.io.FileUtils;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class MasFileResize
{


    public static void main(String args[]) throws IOException
    {
        File whereTheImagesAre = new File("D:\\AllArt");//where is the dataset of images
        String whereToSaveTo = "C:\\Users\\Sean\\Downloads\\ExportImages\\";
        String[] fileTypes = new String[]{ "jpg"};
        int targetWidth = 756;
        int targetHeight = 756;
        boolean convertToGrayscale = true;


         System.out.println("Getting images from " + whereTheImagesAre.getAbsolutePath());

        List<File> listOfFiles =  new ArrayList<>();
        listOfFiles.addAll(FileUtils.listFiles(whereTheImagesAre,fileTypes,true));

        System.out.println("Success. Initiating...");


        int spot = 0;
        for (File image : listOfFiles)
        {
            //Display progress
            if(spot%10 == 0)
            {
                System.out.println();
            }
            System.out.print(".");

            //edit the image
            BufferedImage img = ImageIO.read(image);

            BufferedImage resized = new BufferedImage(targetWidth, targetHeight, img.getType());
            Graphics2D g = resized.createGraphics();
            g.setRenderingHint(RenderingHints.KEY_INTERPOLATION,
                    RenderingHints.VALUE_INTERPOLATION_BILINEAR);
            g.drawImage(img, 0, 0, targetWidth, targetHeight, 0, 0, img.getWidth(),
                    img.getHeight(), null);
            g.dispose();

            img = resized;
            //the image is now resized

            if(convertToGrayscale)
            {
                for (int i = 0; i < targetWidth * targetHeight; i++)
                {

                    //figure out where from the image to get the color of
                    int x = i % targetWidth;
                    int y = i / targetHeight;
                    Color atPixel = new Color(img.getRGB(x, y));

                    int grayValue = (atPixel.getRed()+atPixel.getBlue()+atPixel.getGreen())/3;
                    Color grayColor = new Color(grayValue,grayValue,grayValue);

                    img.setRGB(x,y,grayColor.getRGB());
                }
            }//the image is now grayscale




            //save file
            ImageIO.write(img, "png", new File(whereToSaveTo + spot + ".png"));
            spot++;//increment the spot we are inserting things, so the next image goes in the next spot
        }//end of adding all of the images

    }
}
