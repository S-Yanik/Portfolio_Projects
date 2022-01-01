import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.Stack;

public class FileProcessor
{
    //get all of the files
    //make a stack
    //for each photo
    //  display
    //  crop
    //  fill in with white
    //  shift black white balance
    //  save

    public static JFrame frame;
    public static JPanel panel;
    public static BufferedImage currentImage;
    public static Stack<BufferedImage> bufImageStack;
    public static JLabel imageLabel;


    public static void main(String args[]) throws IOException
    {
        File folder = new File(System.getProperty("user.home") + "/Downloads/TrainingFiles/");
        File[] listOfFiles = folder.listFiles();
        bufImageStack = new Stack<>();

        for (File image : listOfFiles)
        {
            bufImageStack.push(ImageIO.read(image));
        }

        currentImage = bufImageStack.pop();
        BufferedImage other = bufImageStack.pop();

        setupVisuals();

        while(true)
        {
            imageLabel.setIcon(new ImageIcon(currentImage));

        }
    }

    public static void setupVisuals()
    {
        //make a new frame
        frame = new JFrame();
        frame.setTitle("Editor");
        frame.setDefaultCloseOperation(WindowConstants.DISPOSE_ON_CLOSE);
        frame.setMinimumSize(new Dimension(200,200));
        frame.setLayout(new BorderLayout());

        panel = new JPanel();
        imageLabel = new JLabel(new ImageIcon(currentImage));
        panel.add(imageLabel);

        frame.add(panel, BorderLayout.CENTER);

        frame.pack();
        frame.setVisible(true);
    }




}