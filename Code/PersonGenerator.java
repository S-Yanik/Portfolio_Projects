



import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.cpu.nativecpu.NDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.tensorflow.framework.TensorShapeProto;

import javax.imageio.ImageIO;
import javax.swing.*;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;
import javax.swing.filechooser.FileNameExtensionFilter;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.WindowEvent;
import java.awt.event.WindowListener;
import java.awt.image.BufferedImage;
import java.awt.image.ColorModel;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;

public class PersonGenerator
{

    private static JFrame frame;
        private static JPanel topPanel;
            private static ImagePanel mainImagePanel;
            private static JPanel topRightPanel;
                private static JPanel buttonsPanel;
                private static CardLayout mixerHolderLayout;
                private static JPanel mixerHolderPanel;
                    private static JPanel mixerPanel;
                        private static JPanel firstMixFacePanel;
                        private static INDArray firstMixFaceIND;
                        private static JPanel secondMixFacePanel;
                        private static INDArray secondMixFaceIND;
                        private static JPanel combinationMixFacePanel;
                        private static ImagePanel combinationMixFaceImagePanel;
                    private static JPanel emptyMixerPanel;

        private static JPanel bottomPanel;
            private static JPanel featureSliderPanel;
            private static CardLayout sliderPanelLayout;
            private static JPanel featureListPanel;



    private static Color lightGray = new Color(143, 143, 143);
    private static Dimension loadingScreenSize = new Dimension(330, 156);
    private static Dimension defaultFrameSize = new Dimension(Toolkit.getDefaultToolkit().getScreenSize().width * 3 / 4, Toolkit.getDefaultToolkit().getScreenSize().height * 3 / 4);
    private static Dimension buttonDimension = new Dimension(240, 100);

    private static BufferedImage currentFace = new BufferedImage(68, 68, ColorModel.OPAQUE);


    private static int compOutLayerIndex = 1;
    private static int compInLayerIndex = 2;
    private static int lastLayerIndex = 3;
    private static int compDimensions = 90;
    private static INDArray faceValArray = null;
    private static int genWidth = 68;
    private static int genHeight = 68;


    private static MultiLayerNetwork autoEn;


    private static JSlider[] sliderArray = new JSlider[compDimensions];
    private static ArrayList<StoppableListener> listenerList = new ArrayList<>();
    private static int sliderPrecision = 30;


    public static void main(String args[]) throws IOException
    {
        //create frame
        createFrame();
        //display loading screen
        displayLoading();
        //load ai
        autoEn = ModelSerializer.restoreMultiLayerNetwork(System.getProperty("user.home") + "/Downloads/" + "autoSave" + ".zip");
        setFaceValarrayToDefault();
        //display final frame
        displayProper();
    }


    public static void createFrame()
    {
        frame = new JFrame();
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setResizable(true);//Sure, why not resize it
    }

    public static void displayLoading()
    {
        frame.setUndecorated(true);
        frame.setPreferredSize(loadingScreenSize);
        frame.setLocation(getMiddleOfScreen(loadingScreenSize));

        JLabel loadingLabel = new JLabel("Loading");
        loadingLabel.setOpaque(true);
        loadingLabel.setBackground(Color.BLACK);
        loadingLabel.setFont(new Font("", Font.BOLD, 35));
        loadingLabel.setForeground(Color.white);
        loadingLabel.setHorizontalAlignment(JLabel.CENTER);


        frame.getContentPane().add(loadingLabel, BorderLayout.CENTER);

        JLabel waitAWhileLabel = new JLabel("(This could take some time on older devices)");
        waitAWhileLabel.setOpaque(true);
        waitAWhileLabel.setBackground(new Color(35, 35, 35));
        waitAWhileLabel.setFont(new Font("", Font.PLAIN, 15));
        waitAWhileLabel.setForeground(Color.white);
        waitAWhileLabel.setHorizontalAlignment(JLabel.CENTER);


        frame.getContentPane().add(waitAWhileLabel, BorderLayout.PAGE_END);

        frame.pack();
        frame.setVisible(true);
    }


    /*
      private static JFrame frame;
        private static JPanel topPanel;
            private static ImagePanel mainImagePanel;
            private static JPanel topRightPanel;
                private static JPanel buttonsPanel;
        private static JPanel bottomPanel;
            private static FeatureSliderPanel featureSliderPanel;
            private static FeatureListPanel featureListPanel;
     */
    public static void displayProper() throws IOException
    {
        frame.removeAll();
        frame.dispose();

        frame = new JFrame();
        frame.setVisible(false);
        frame.setUndecorated(false);
        frame.setPreferredSize(defaultFrameSize);
        frame.setLocation(getMiddleOfScreen(defaultFrameSize));
        frame.setTitle("Face Generator");
        frame.setBackground(Color.DARK_GRAY);
        frame.getContentPane().setBackground(Color.DARK_GRAY);
        generateTopFrame();
        generateBottomFrame();

        frame.pack();
        frame.setVisible(true);

        frame.addWindowListener(new MyWindowListener());
    }

    public static void generateBottomFrame()
    {
        bottomPanel = new JPanel(new BorderLayout());
        bottomPanel.setOpaque(true);
        bottomPanel.setBackground(Color.DARK_GRAY);
        generateFeatureSliderPanel();
        bottomPanel.add(featureSliderPanel, BorderLayout.WEST);
        generateFeatureList();
        bottomPanel.add(featureListPanel, BorderLayout.EAST);

        frame.getContentPane().add(bottomPanel, BorderLayout.SOUTH);
    }

    public static void generateFeatureSliderPanel()
    {

        sliderPanelLayout = new CardLayout();
        featureSliderPanel = new JPanel(sliderPanelLayout);
        featureSliderPanel.setOpaque(true);
        featureSliderPanel.setBackground(Color.DARK_GRAY);


        featureSliderPanel.add("empty", generateEmptyPanel());
        featureSliderPanel.add("emotion",generateEmotionPanel());
        featureSliderPanel.add("eyes",generateEyesPanel());
       featureSliderPanel.add("gender",generateGenderPanel());
       featureSliderPanel.add("features",generateFeaturesPanel());
       featureSliderPanel.add("head",generateHeadPanel());
        featureSliderPanel.add("jaw", generateJawPanel());
       featureSliderPanel.add("light",generateLightPanel());
       featureSliderPanel.add("rot",generateRotPanel());




    }

    public static JPanel generateRotPanel()
    {
        JPanel panel = new JPanel(new GridLayout(1,10,5,20));
        panel.setBackground(lightGray);
        panel.setOpaque(true);


        panel.add(generateLabeledFeatureSlider("Up/Down",9));
        panel.add(generateLabeledFeatureSlider("Up/Down",16));
        panel.add(generateLabeledFeatureSlider("Up/Down",25));
        panel.add(generateLabeledFeatureSlider("Up/Down",28));
        panel.add(generateLabeledFeatureSlider("Top of Head",33));
        panel.add(generateLabeledFeatureSlider("Asymmetry",12));
        panel.add(generateLabeledFeatureSlider("Up/Down",48));
        panel.add(generateLabeledFeatureSlider("Up/Down",51));
        panel.add(generateLabeledFeatureSlider("Left/Right",70));
        panel.add(generateLabeledFeatureSlider("Multi",75));


        return panel;
    }

    public static JPanel generateLightPanel()
    {
        JPanel panel = new JPanel(new GridLayout(1,18,5,20));
        panel.setBackground(lightGray);
        panel.setOpaque(true);


        panel.add(generateLabeledFeatureSlider("Lighting",49));
        panel.add(generateLabeledFeatureSlider("Diffuse",50));
        panel.add(generateLabeledFeatureSlider("Lighting",55));
        panel.add(generateLabeledFeatureSlider("Diffuse",56));
        panel.add(generateLabeledFeatureSlider("Diffuse",58));
        panel.add(generateLabeledFeatureSlider("Lighting",63));
        panel.add(generateLabeledFeatureSlider("Diffuse",68));
        panel.add(generateLabeledFeatureSlider("Lighting",69));
        panel.add(generateLabeledFeatureSlider("Brightness",83));
        panel.add(generateLabeledFeatureSlider("Spotlight",90));
        panel.add(generateLabeledFeatureSlider("Blur",73));
        panel.add(generateLabeledFeatureSlider("Clarity",86));
        panel.add(generateLabeledFeatureSlider("Diffuse",2));
        panel.add(generateLabeledFeatureSlider("Above",26));
        panel.add(generateLabeledFeatureSlider("Diffuse",27));
        panel.add(generateLabeledFeatureSlider("Blurriness",44));
        panel.add(generateLabeledFeatureSlider("Contrast",11));
        panel.add(generateLabeledFeatureSlider("Contrast",37));


        return panel;
    }

    public static JPanel generateHeadPanel()
    {
        JPanel panel = new JPanel(new GridLayout(1,15,5,20));
        panel.setBackground(lightGray);
        panel.setOpaque(true);


        panel.add(generateLabeledFeatureSlider("Fullness",61));
        panel.add(generateLabeledFeatureSlider("Width",66));
        panel.add(generateLabeledFeatureSlider("Neck",82));
        panel.add(generateLabeledFeatureSlider("Forehead",89));
        panel.add(generateLabeledFeatureSlider("Size",1));
        panel.add(generateLabeledFeatureSlider("Size",5));
        panel.add(generateLabeledFeatureSlider("Vertical",17));
        panel.add(generateLabeledFeatureSlider("Fullness",18));
        panel.add(generateLabeledFeatureSlider("Width",19));
        panel.add(generateLabeledFeatureSlider("Width",47));
        panel.add(generateLabeledFeatureSlider("Head Edge",30));



        return panel;
    }

    public static JPanel generateFeaturesPanel()
    {
        JPanel panel = new JPanel(new GridLayout(1,15,5,20));
        panel.setBackground(lightGray);
        panel.setOpaque(true);


        panel.add(generateLabeledFeatureSlider("Cheeks",10));
        panel.add(generateLabeledFeatureSlider("Cheeks",15));
        panel.add(generateLabeledFeatureSlider("Eyelids",21));
        panel.add(generateLabeledFeatureSlider("Philitrum",22));
        panel.add(generateLabeledFeatureSlider("Nose",23));
        panel.add(generateLabeledFeatureSlider("Flatness",38));
        panel.add(generateLabeledFeatureSlider("Paleness",35));
        panel.add(generateLabeledFeatureSlider("Skintone",45));
        panel.add(generateLabeledFeatureSlider("Nose",85));
        panel.add(generateLabeledFeatureSlider("Nose",87));
        panel.add(generateLabeledFeatureSlider("Nose Type",80));
        panel.add(generateLabeledFeatureSlider("Lips",79));


        return panel;
    }

    public static JPanel generateGenderPanel()
    {
        JPanel panel = new JPanel(new GridLayout(1,15,5,20));
        panel.setBackground(lightGray);
        panel.setOpaque(true);


        panel.add(generateLabeledFeatureSlider("Eyebrows",52));
        panel.add(generateLabeledFeatureSlider("Eyebrows",65));
        panel.add(generateLabeledFeatureSlider("Head",78));
        panel.add(generateLabeledFeatureSlider("Jaw",3));
        panel.add(generateLabeledFeatureSlider("Head",31));
        panel.add(generateLabeledFeatureSlider("Eyes",62));
        panel.add(generateLabeledFeatureSlider("Gender",64));
        panel.add(generateLabeledFeatureSlider("Gender",34));

        return panel;
    }

    public static JPanel generateEmotionPanel()
    {
        JPanel panel = new JPanel(new GridLayout(1,15,5,20));
        panel.setBackground(lightGray);
        panel.setOpaque(true);


        panel.add(generateLabeledFeatureSlider("Teeth",54));
        panel.add(generateLabeledFeatureSlider("Smile",57));
        panel.add(generateLabeledFeatureSlider("Mood",67));
        panel.add(generateLabeledFeatureSlider("Teeth",53));
        panel.add(generateLabeledFeatureSlider("Mood",60));
        panel.add(generateLabeledFeatureSlider("Mood",74));
        panel.add(generateLabeledFeatureSlider("Mood",71));
        panel.add(generateLabeledFeatureSlider("Energy",72));
        panel.add(generateLabeledFeatureSlider("Dark",88));
        panel.add(generateLabeledFeatureSlider("Joy",13));
        panel.add(generateLabeledFeatureSlider("Intensity",14));
        panel.add(generateLabeledFeatureSlider("Eyes",20));
        panel.add(generateLabeledFeatureSlider("Frown",41));
        panel.add(generateLabeledFeatureSlider("Frown",43));
        panel.add(generateLabeledFeatureSlider("Teeth",8));




        return panel;
    }

    public static JPanel generateEyesPanel()
    {
        JPanel panel = new JPanel(new GridLayout(1,15,5,20));
        panel.setBackground(lightGray);
        panel.setOpaque(true);


        panel.add(generateLabeledFeatureSlider("Glow",6));
        panel.add(generateLabeledFeatureSlider("Size",24));
        panel.add(generateLabeledFeatureSlider("Visible",42));
        panel.add(generateLabeledFeatureSlider("Shape",46));
        panel.add(generateLabeledFeatureSlider("Towards",32));
        panel.add(generateLabeledFeatureSlider("Shape",59));
        panel.add(generateLabeledFeatureSlider("Towards",76));
        panel.add(generateLabeledFeatureSlider("Height",77));
        panel.add(generateLabeledFeatureSlider("Size",81));
        panel.add(generateLabeledFeatureSlider("Shift",84));

        return panel;
    }
    public static JPanel generateEmptyPanel()
    {
        JPanel emptyPanel = new JPanel();
        emptyPanel.setOpaque(true);
        emptyPanel.setBackground(Color.DARK_GRAY);
        return emptyPanel;
    }

    public static JPanel generateJawPanel()
    {
        JPanel jawPanel = new JPanel(new GridLayout(1,5,40,20));
        jawPanel.setBackground(lightGray);
        jawPanel.setOpaque(true);


        jawPanel.add(generateLabeledFeatureSlider("Height",7));
        jawPanel.add(generateLabeledFeatureSlider("Shape",29));
        jawPanel.add(generateLabeledFeatureSlider("Tilt",36));
        jawPanel.add(generateLabeledFeatureSlider("Shape",39));
        jawPanel.add(generateLabeledFeatureSlider("Height",40));

        return jawPanel;
    }


    public static JPanel generateLabeledFeatureSlider(String name, int index)
    {
        JPanel panel = new JPanel(new BorderLayout());
        panel.setOpaque(true);
        panel.setBackground(lightGray);
        panel.add(new SLabel(name),BorderLayout.NORTH);
        panel.add(generateFeatureSlider(index),BorderLayout.SOUTH);
        return panel;
    }

    public static class SLabel extends JLabel
    {
        public Dimension size = new Dimension(40,12);
        public SLabel(String name)
        {
            super (name);
            this.setBackground(Color.BLACK);
            this.setFont(new Font("",Font.BOLD,10));
            this.setForeground(Color.white);
        }

        public Dimension getPreferredSize()
        {
            return size;
        }
    }

    //indexes start at 1!!! not zero! Sorry
    public static JSlider generateFeatureSlider(int index)
    {
        index = index-1;
        float mean = (float) deviationsAndMeans[index][1];
        float standardD = (float) deviationsAndMeans[index][0];
        float minValue = mean - standardD*2.5f;
        float maxValue = mean + standardD*2.5f;
        final int precision = sliderPrecision;

        final JSlider slider = new JSlider(JSlider.VERTICAL, (int)(minValue*precision), (int)(maxValue * precision), (int)(mean*precision))
        {
            public Dimension getPreferredSize()
            {
                return new Dimension(20,320);
            }
        };
        final int finalIndex = index;
        StoppableListener listener = new StoppableListener()
        {
            public void stateChanged(ChangeEvent e)//when clicked
            {
                if (active)
                {
                    float calculatedValue = slider.getValue() / precision;
                    changeFaceValIndex(finalIndex, calculatedValue);
                }
            }
        };
        slider.addChangeListener(listener);
        listenerList.add(listener);
        slider.setMajorTickSpacing(30);
        slider.setPaintTicks(true);
        slider.setBackground(Color.black);

        sliderArray[index] = slider;
        return slider;
    }

    public static void generateFeatureList()
    {
        featureListPanel = new JPanel(new GridLayout(3, 2, 10, 10))
        {
            public Dimension getPreferredSize()
            {
                return new Dimension((int)(defaultFrameSize.width / 2.8), defaultFrameSize.height / 2 - 50);
            }
        };
        featureListPanel.setOpaque(true);
        featureListPanel.setBackground(new Color(143, 143, 143));


        Dimension buttonSize = new Dimension(60,30);

        SButton emotionLink = new SButton("Emotion");
        emotionLink.size = buttonSize;
        emotionLink.addActionListener(new ActionListener()
        {
            public void actionPerformed(ActionEvent e)//when clicked
            {
                synchronized (currentFace)
                {
                sliderPanelLayout.show(featureSliderPanel,"emotion");
                }
            }
        });
        featureListPanel.add(emotionLink);

        SButton eyesLink = new SButton("Eyes");
        eyesLink.size = buttonSize;
        eyesLink.addActionListener(new ActionListener()
        {
            public void actionPerformed(ActionEvent e)//when clicked
            {
                synchronized (currentFace)
                {
                    sliderPanelLayout.show(featureSliderPanel,"eyes");
                }
            }
        });
        featureListPanel.add(eyesLink);

        SButton genderLink = new SButton("Gender");
        genderLink.size = buttonSize;
        genderLink.addActionListener(new ActionListener()
        {
            public void actionPerformed(ActionEvent e)//when clicked
            {
                synchronized (currentFace)
                {
                    sliderPanelLayout.show(featureSliderPanel,"gender");
                }
            }
        });
        featureListPanel.add(genderLink);

        SButton featuresLink = new SButton("Key Features");
        featuresLink.size = buttonSize;
        featuresLink.addActionListener(new ActionListener()
        {
            public void actionPerformed(ActionEvent e)//when clicked
            {
                synchronized (currentFace)
                {
                    sliderPanelLayout.show(featureSliderPanel,"features");
                }
            }
        });
        featureListPanel.add(featuresLink);

        SButton headLink = new SButton("Head Shape");
        headLink.size = buttonSize;
        headLink.addActionListener(new ActionListener()
        {
            public void actionPerformed(ActionEvent e)//when clicked
            {
                synchronized (currentFace)
                {
                    sliderPanelLayout.show(featureSliderPanel,"head");
                }
            }
        });
        featureListPanel.add(headLink);

        SButton jawLink = new SButton("Jaw");
        jawLink.size = buttonSize;
        jawLink.addActionListener(new ActionListener()
        {
            public void actionPerformed(ActionEvent e)//when clicked
            {
                synchronized (currentFace)
                {
                    sliderPanelLayout.show(featureSliderPanel,"jaw");
                }
            }
        });
        featureListPanel.add(jawLink);

        SButton lightingLink = new SButton("Lighting");
        lightingLink.size = buttonSize;
        lightingLink.addActionListener(new ActionListener()
        {
            public void actionPerformed(ActionEvent e)//when clicked
            {
                synchronized (currentFace)
                {
                    sliderPanelLayout.show(featureSliderPanel,"light");
                }
            }
        });
        featureListPanel.add(lightingLink);



        SButton rotLink = new SButton("Rotation");
        rotLink.size = buttonSize;
        rotLink.addActionListener(new ActionListener()
        {
            public void actionPerformed(ActionEvent e)//when clicked
            {
                synchronized (currentFace)
                {
                    sliderPanelLayout.show(featureSliderPanel,"rot");
                }
            }
        });
        featureListPanel.add(rotLink);

        SButton closeLink = new SButton("Close All");
        closeLink.size = buttonSize;
        closeLink.addActionListener(new ActionListener()
        {
            public void actionPerformed(ActionEvent e)//when clicked
            {
                synchronized (currentFace)
                {
                    sliderPanelLayout.show(featureSliderPanel,"empty");
                }
            }
        });
        featureListPanel.add(closeLink);


    }



    public static void generateTopFrame() throws IOException
    {
        topPanel = new JPanel(new BorderLayout());
        topPanel.setOpaque(true);
        topPanel.setBackground(Color.DARK_GRAY);
        mainImagePanel = new ImagePanel();
        mainImagePanel.scale = 3d;
        mainImagePanel.setOpaque(true);
        mainImagePanel.setBackground(Color.darkGray);
        mainImagePanel.isMainImage = true;
       setFaceValarrayToDefault();

        topPanel.add(mainImagePanel, BorderLayout.WEST);

        topRightPanel = new JPanel(new BorderLayout());
        topRightPanel.setOpaque(true);
        topRightPanel.setBackground(Color.DARK_GRAY);
        generateButtonsPanel();
        topRightPanel.add(buttonsPanel, BorderLayout.NORTH);
        generateMixerHolderPanel();
        topRightPanel.add(mixerHolderPanel,BorderLayout.SOUTH);
        topPanel.add(topRightPanel, BorderLayout.EAST);


        frame.getContentPane().add(topPanel, BorderLayout.NORTH);
    }

    public static void generateMixerHolderPanel()
    {
        mixerHolderLayout = new CardLayout();
        mixerHolderPanel = new JPanel(mixerHolderLayout);

        emptyMixerPanel = new JPanel();
        emptyMixerPanel.setOpaque(true);
        emptyMixerPanel.setBackground(Color.darkGray);
        mixerHolderPanel.add("blank",emptyMixerPanel);


        //@todo generate mixer panel
        mixerPanel = new JPanel()
        {
            public Dimension getPreferredSize()
            {
                return  new Dimension(500,270);
            }
        };
        mixerPanel.setOpaque(true);
        mixerPanel.setBackground(new Color(143, 143, 143));
        mixerPanel.setLayout( new GridLayout(1,3,20,20));




        firstMixFacePanel = new JPanel();
        firstMixFacePanel.setOpaque(true);
        firstMixFacePanel.setBackground(new Color(143, 143, 143));




        final ImagePanel firstFaceImagePanel = new ImagePanel();
        firstFaceImagePanel.displayImage = fullINDtoBuffImg(compINDToFullIND(getDefaultFaceArray()));
        firstFaceImagePanel.scale = 2d;
        firstMixFaceIND = getDefaultFaceArray();
        firstMixFacePanel.add(firstFaceImagePanel);



        SButton firstFaceLoadButton = new SButton("Select First Face");
        firstFaceLoadButton.addActionListener(new ActionListener()
        {
            public void actionPerformed(ActionEvent e)//when clicked
            {
                try
                {
                    INDArray compressed = buffImageToCompIND(haveUserLoadAnImage());
                    firstMixFaceIND = compressed;
                    firstFaceImagePanel.displayImage = fullINDtoBuffImg(compINDToFullIND(firstMixFaceIND));
                    firstFaceImagePanel.repaint();
                    combinationMixFaceImagePanel.displayImage = fullINDtoBuffImg(compINDToFullIND(averageTwoINDArrays(firstMixFaceIND,secondMixFaceIND)));
                    combinationMixFaceImagePanel.repaint();
                } catch (Exception ex)
                {
                    System.out.println("An error occurred when trying to load an image. Did they not select anything?");
                    ex.printStackTrace();
                }
            }
        });
        firstMixFacePanel.add(firstFaceLoadButton);


        mixerPanel.add(firstMixFacePanel);





        secondMixFacePanel = new JPanel();
        secondMixFacePanel.setOpaque(true);
        secondMixFacePanel.setBackground(new Color(143, 143, 143));



        final ImagePanel secondFaceImagePanel = new ImagePanel();
        secondFaceImagePanel.displayImage = fullINDtoBuffImg(compINDToFullIND(getDefaultFaceArray()));
        secondFaceImagePanel.scale = 2d;
        secondMixFaceIND = getDefaultFaceArray();
        secondMixFacePanel.add(secondFaceImagePanel);



        SButton secondFaceLoadButton = new SButton("Select Second Face");
        secondFaceLoadButton.addActionListener(new ActionListener()
        {
            public void actionPerformed(ActionEvent e)//when clicked
            {
                try
                {
                    INDArray compressed = buffImageToCompIND(haveUserLoadAnImage());
                    secondMixFaceIND = compressed;
                    secondFaceImagePanel.displayImage = fullINDtoBuffImg(compINDToFullIND(secondMixFaceIND));
                    secondFaceImagePanel.repaint();
                    combinationMixFaceImagePanel.displayImage = fullINDtoBuffImg(compINDToFullIND(averageTwoINDArrays(firstMixFaceIND,secondMixFaceIND)));
                    combinationMixFaceImagePanel.repaint();
                } catch (Exception ex)
                {
                    System.out.println("An error occurred when trying to load an image. Did they not select anything?");
                    ex.printStackTrace();
                }
            }
        });

        secondMixFacePanel.add(secondFaceLoadButton);

        mixerPanel.add(secondMixFacePanel);



        combinationMixFacePanel = new JPanel();
        combinationMixFacePanel.setOpaque(true);
        combinationMixFacePanel.setBackground(new Color(143, 143, 143));


        combinationMixFaceImagePanel = new ImagePanel();
        combinationMixFaceImagePanel.displayImage = fullINDtoBuffImg(compINDToFullIND(getDefaultFaceArray()));
        combinationMixFaceImagePanel.scale = 2d;
        combinationMixFacePanel.add(combinationMixFaceImagePanel);

        SButton setFaceToMixButton = new SButton("Set as Main Face");
        setFaceToMixButton.addActionListener(new ActionListener()
        {
            public void actionPerformed(ActionEvent e)//when clicked
            {
                try
                {
                    INDArray averagedFace = averageTwoINDArrays(firstMixFaceIND,secondMixFaceIND);
                    setFaceValarray(averagedFace);
                } catch (Exception ex)
                {
                    System.out.println("An error occurred when trying to load an image. Did they not select anything?");
                    ex.printStackTrace();
                }
            }
        });

        combinationMixFacePanel.add(setFaceToMixButton);

        mixerPanel.add(combinationMixFacePanel);

        mixerHolderPanel.add("mix",mixerPanel);
    }

    public static void generateButtonsPanel()
    {
        buttonsPanel = new JPanel();
        buttonsPanel.setOpaque(true);
        buttonsPanel.setBackground(Color.DARK_GRAY);

        //buttons:
        //Save ✔
        //Load ✔
        //Random ✔
        //Mixer
        //Customise


        SButton saveButton = new SButton("Save to PNG");
        saveButton.addActionListener(new ActionListener()
        {
            public void actionPerformed(ActionEvent e)//when clicked
            {
                synchronized (currentFace)
                {
                    BufferedImage toSave = currentFace;


                    //makes sure it doesn't accidentally write over an already saved file
                    //will become slow if you save hundreds (thousands) of files in one session, but that could be fixed with extreme ease
                    //I'm not going to to avoid clutter
                    boolean generatedAFileName = false;
                    File location = null;
                    int spot = -1;
                    while (!generatedAFileName)
                    {
                        spot++;
                        location = new File(System.getProperty("user.home") + "/Downloads/" + "face" + spot + ".PNG");
                        if (!location.exists())//sees if a file has already been generated here. If not, use that spot for the file
                        {
                            generatedAFileName = true;
                        }
                    }


                    try
                    {
                        System.out.println("saving");
                        ImageIO.write(toSave, "PNG", location);
                        System.out.print("Saved");
                    } catch (IOException ex)
                    {
                        ex.printStackTrace();
                    }
                }
            }
        });
        buttonsPanel.add(saveButton);


        SButton loadButton = new SButton("Load Image");
        loadButton.addActionListener(new ActionListener()
        {
            public void actionPerformed(ActionEvent e)//when clicked
            {
                try
                {
                    INDArray compressed = buffImageToCompIND(haveUserLoadAnImage());
                    setFaceValarray(compressed);
                } catch (Exception ex)
                {
                    System.out.println("An error occurred when trying to load an image. Did they not select anything?");
                    ex.printStackTrace();
                }
            }
        });
        buttonsPanel.add(loadButton);





        SButton randomButton = new SButton("Random Face");
        randomButton.addActionListener(new ActionListener()
        {
            public void actionPerformed(ActionEvent e)//when clicked
            {
                float[] randomFloats = new float[compDimensions];
                for (int i = 0; i < compDimensions; i++)
                {
                    //a random range between DISTANCE standard deviations of the mean
                    float dist = 0.6f;//could be 4 standard deviations, or 0.5, or any number
                    float mean = (float) deviationsAndMeans[i][1];
                    float stndDev = (float) deviationsAndMeans[i][0];

                    randomFloats[i] = (float) (mean - (stndDev * dist) + Math.random() * 2 * dist * stndDev);
                }

                setFaceValarray(new NDArray(randomFloats));
            }
        });
        buttonsPanel.add(randomButton);





        final SButton mixerButton = new SButton("Open Mixing Menu");
        mixerButton.addActionListener(new ActionListener()
        {
            public void actionPerformed(ActionEvent e)//when clicked
            {
                mixerHolderLayout.next(mixerHolderPanel);
            }
        });
        buttonsPanel.add(mixerButton);


    }//end of generateButtons


    //given the size of a frame, if we want to have that frame in the middle of the screen, where is the
    //top left point?
    private static Point getMiddleOfScreen(Dimension frameSize)
    {
        Point location = new Point();
        location.x = (Toolkit.getDefaultToolkit().getScreenSize().width - frameSize.width) / 2;
        location.y = (Toolkit.getDefaultToolkit().getScreenSize().height - frameSize.height) / 2;

        return location;
    }

    static class MyWindowListener implements WindowListener
    {

        public void windowClosing(WindowEvent arg0) {
            System.exit(0);
        }

        public void windowOpened(WindowEvent arg0) {}
        public void windowClosed(WindowEvent arg0) {}
        public void windowIconified(WindowEvent arg0) {}
        public void windowDeiconified(WindowEvent arg0) {}
        public void windowActivated(WindowEvent arg0) {}
        public void windowDeactivated(WindowEvent arg0) {}

    }

    private static void changeFaceValIndex(int index, float value)
    {
        synchronized (faceValArray)
        {
            if(faceValArray.getFloat(index) != value)
            {
                float[] newArray = new float[compDimensions];
                for (int i = 0; i < compDimensions; i++)
                {
                    if (i != index)
                    {
                        newArray[i] = faceValArray.getFloat(i);
                    } else
                    {
                        newArray[i] = value;
                    }
                }
                faceValArray = new NDArray(newArray);

                regenerateFace();
            }
        }
    }
    //returns a compressed INDArray
    private static INDArray buffImageToCompIND(BufferedImage img)
    {

        img = scaleImage(img, (double) (genWidth) / img.getWidth(), (double) (genHeight) / img.getHeight());

        float[] imageFloats = new float[genWidth * genHeight * 3];


        //get all the actual values from the image
        for (int i = 0; i < genWidth * genHeight; i++)
        {

            //figure out where from the image to get the color of
            int x = i % genWidth;
            int y = i / genHeight;
            Color atPixel = new Color(img.getRGB(x, y));


            //get the red green and blue values, add to our array
            imageFloats[i] = atPixel.getRed() / 255f;
            imageFloats[i + genWidth * genHeight] = atPixel.getGreen() / 255f;//
            imageFloats[i + genWidth * genHeight * 2] = atPixel.getBlue() / 255f;

        }
        INDArray convertedFloats = new NDArray(imageFloats);

        return autoEn.activateSelectedLayers(0, compOutLayerIndex, convertedFloats);
    }

    private static BufferedImage haveUserLoadAnImage()
    {

        JFileChooser chooser = new JFileChooser();
        FileNameExtensionFilter filter = new FileNameExtensionFilter(
                "Image Files", "jpg", "JPG", "png", "PNG", "jpeg");
        chooser.setFileFilter(filter);
        chooser.setCurrentDirectory(new File(System.getProperty("user.home") + "\\Downloads"));
        int returnVal = chooser.showOpenDialog(null);
        if (returnVal == JFileChooser.APPROVE_OPTION)
        {//if they chose a file

            try
            {
                return (ImageIO.read(chooser.getSelectedFile()));
            } catch (IOException ex)
            {
                System.out.println("An error occurred when trying to read a file that the user selected.");
                ex.printStackTrace();
            }
        }

        return null;
    }


    public static INDArray compINDToFullIND(INDArray compressed)
    {
        return autoEn.activateSelectedLayers(compInLayerIndex, lastLayerIndex, compressed);
    }

    public static BufferedImage fullINDtoBuffImg(INDArray inputArray)
    {
        int width = genWidth;
        int height = genHeight;
        //the image to return (B uffered I mage --> bi). Sorry, I guess that's pretty obvious.
        BufferedImage bi = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);

        //go through the 1/3 of the INDArray (the array is width * height for red, green and blue, so just width
        // times height is 1/3 of it)
        for (int i = 0; i < width * height; i++)//This turns the array into our image
        {
            //calculate where the x and y values should be on the actual image
            int x = i % width;
            int y = i / width;

            //find the reds, greens, and blues
            int red = (int) (255 * inputArray.getFloat(i)); //get the red directly from the spot
            int green = (int) (255 * inputArray.getFloat(i + width * height)); //get the green from the spot 1/3 of the array forwards
            int blue = (int) (255 * inputArray.getFloat(i + width * height * 2)); //get the blue from the spot 2/3 of the array forwards

            //There is a possibility that the AI has generated impossible colors
            //(IE darker than black (RGB = -20, -5, 0, for example) or brighter than white
            //clamp these values to get something decent (clamping forces the values to be between the min and max)
            red = clamp(red, 0, 255);
            green = clamp(green, 0, 255);
            blue = clamp(blue, 0, 255);

            //get the color for the pixel (stored as an int)
            int colorOfPixel = new Color(red, green, blue).getRGB();

            //set the color at the spot
            bi.setRGB(x, y, colorOfPixel);

        }//end of coloring in our buffered image. It is now correctly made

        return bi;
    }

    //Scales a buffered image
    private static BufferedImage scaleImage(BufferedImage image, double xScale, double yScale)
    {
        int width = image.getWidth();
        int height = image.getHeight();

        Image imageScaled = null;
        if (xScale > 1 && yScale > 1)
        {
            imageScaled = image.getScaledInstance((int) (xScale * width), (int) (yScale * height), Image.SCALE_REPLICATE);
        } else
        {
            imageScaled = image.getScaledInstance((int) (xScale * width), (int) (yScale * height), Image.SCALE_REPLICATE);
        }

        BufferedImage bimage = new BufferedImage(imageScaled.getWidth(null), imageScaled.getHeight(null), BufferedImage.TYPE_INT_RGB);

        // Draw the image on to the buffered image
        Graphics2D bGr = bimage.createGraphics();
        bGr.drawImage(imageScaled, 0, 0, null);
        bGr.dispose();

        return bimage;
    }


    public static void updateFaceImage(BufferedImage newImage)
    {
        synchronized (currentFace)
        {
            currentFace = newImage;
            mainImagePanel.displayImage = currentFace;
        }

        mainImagePanel.repaint();
    }

    public static void setFaceValarray(INDArray newValArray)
    {
        synchronized (faceValArray)
        {
            faceValArray = newValArray;

            for (StoppableListener listener : listenerList)
            {
                listener.setActive(false);
            }

            for (int i = 0; i < compDimensions; i++)
            {
                JSlider currentSlider = sliderArray[i];

                if (currentSlider != null)
                {
                    currentSlider.setValue((int)(newValArray.getFloat(i)*sliderPrecision));
                }
            }

            for (StoppableListener listener : listenerList)
            {
                listener.setActive(true);
            }
        }

        regenerateFace();
    }

    public static void setFaceValarrayToDefault()
    {
        INDArray defaultValuesAsIND = getDefaultFaceArray();

            if (faceValArray != null)
            {
                setFaceValarray(defaultValuesAsIND);
            }
            else
            {
                faceValArray = defaultValuesAsIND;//if the faceValArray was never initialized
            }
    }

    public static INDArray getDefaultFaceArray()
    {
        float[] defaultValues = new float[compDimensions];
        for (int i = 0; i < compDimensions; i++)
        {
            defaultValues[i] = (float) deviationsAndMeans[i][1];
        }

        return new NDArray(defaultValues);
    }


    public static void regenerateFace()
    {
        BufferedImage newFace;
        synchronized (faceValArray)
        {
            newFace = fullINDtoBuffImg(compINDToFullIND(faceValArray));
        }

        updateFaceImage(newFace);
    }



    public static void updateFaceScale(double newScale)
    {
        synchronized (mainImagePanel.scale)
        {
            mainImagePanel.scale = newScale;
        }
    }

    public static INDArray averageTwoINDArrays(INDArray first, INDArray second)
    {
        float[] combined = new float[(int) first.length()];

        for (int i = 0; i < combined.length; i++)
        {
            combined[i] = (first.getFloat(i) + second.getFloat(i))/2;
        }

        return new NDArray(combined);
    }








    //clamps value between min and max
    public static int clamp(int value, int min, int max)
    {
        return Math.max(min,Math.min(max,value));
    }












    //Panels



    public static class ImagePanel extends JPanel
    {
        public BufferedImage displayImage;
        public Double scale;
        public int padding = 10;
        public int imageBorder = 3;
        public boolean isMainImage = false;

        public ImagePanel()
        {
            this.setOpaque(true);
        }

        public void paint(Graphics g)
        {
            g.setColor(Color.DARK_GRAY);
            g.fillRect(0, 0, (int)(displayImage.getWidth()*scale)+padding*2,(int)(displayImage.getHeight()*scale)+padding*2);

            if (isMainImage)
            {
                g.setColor(Color.DARK_GRAY);
                g.fillRect(0,0,500,500);
            }

            g.setColor(Color.lightGray);
            g.fillRect(padding-imageBorder,padding-imageBorder,(int)(displayImage.getWidth()*scale)+imageBorder*2,(int)(displayImage.getHeight()*scale)+imageBorder*2);
            g.drawImage(displayImage, padding, padding,  (int)(displayImage.getWidth()*scale),(int)(displayImage.getHeight()*scale),null);
        }

        public Dimension getPreferredSize()
        {
         return new Dimension((int)(displayImage.getWidth()*scale)+padding*2,(int)(displayImage.getHeight()*scale)+padding*2);
        }
    }

    public static class SButton extends JButton
    {
        public Dimension size = buttonDimension;
        public SButton(String name)
        {
            super (name);
            this.setBackground(Color.BLACK);
            this.setFont(new Font("",Font.BOLD,20));
            this.setForeground(Color.white);
        }

        public Dimension getPreferredSize()
            {
                return size;
            }
    }

    private static double[][] deviationsAndMeans =
            {
                    {1.3184291,  -1.9635012},
                    {1.1143454,  1.0607489},
                    {1.2812743,  -2.122794},
                    {1.3293965,  1.6425794},
                    {1.1263304,  -3.2390466},
                    {1.6874282,  0.06793474},
                    {1.716432,  0.37785536},
                    {1.0864205,  1.6584553},
                    {1.4057993,  2.780536},
                    {1.2811594,  1.4840785},
                    {1.3185261,  1.6291887},
                    {1.73875,  -0.12967883},
                    {1.1510271,  -1.6093639},
                    {1.3526037,  -1.6212988},
                    {1.2437602,  2.3040082},
                    {1.2130065,  -0.90853804},
                    {1.2880337,  -0.033746764},
                    {1.1923515,  -2.2381063},
                    {1.1679782,  1.455376},
                    {1.6408468,  1.0245857},
                    {1.3102198,  -0.99530506},
                    {1.1510891,  -0.7344639},
                    {1.5110672,  2.505438},
                    {1.2065122,  -2.230175},
                    {1.4614555,  -2.9804316},
                    {1.5096477,  -4.342641},
                    {1.1449405,  -2.447658},
                    {1.0552062,  1.6544958},
                    {1.1828862,  -0.479899},
                    {1.1497827,  -1.0687492},
                    {1.0867883,  1.8851519},
                    {1.5342823,  0.29908672},
                    {1.171479,  1.4958943},
                    {1.6400094,  0.31288353},
                    {1.290808,  1.1482807},
                    {1.0930868,  -0.5246228},
                    {1.3630533,  1.7202567},
                    {1.1733103,  1.8254694},
                    {1.0152318,  -1.3248806},
                    {1.6251496,  1.1055925},
                    {1.3398885,  3.936369},
                    {1.1295586,  -0.6981749},
                    {1.3908918,  0.60767627},
                    {1.3754667,  3.127226},
                    {1.2303625,  1.8460565},
                    {1.2198272,  2.9385817},
                    {1.3069615,  -2.439613},
                    {1.2440125,  1.6893052},
                    {1.518797,  0.16196282},
                    {1.1206629,  -2.9032397},
                    {1.552014,  -0.6720307},
                    {1.1646677,  0.5692929},
                    {1.4481382,  -0.18407293},
                    {1.3421469,  1.482679},
                    {1.1233813,  1.0214801},
                    {1.1891111,  0.5162861},
                    {1.5095499,  -0.6994655},
                    {1.0447835,  0.86065024},
                    {1.0893226,  -1.9598515},
                    {1.4958693,  -0.46305338},
                    {1.0425583,  -1.6664437},
                    {1.5003065,  -1.3227804},
                    {1.4937384,  0.43166125},
                    {1.4257604,  -0.48177853},
                    {1.0815598,  -0.806296},
                    {1.2310427,  1.3988352},
                    {1.6545726,  0.9063628},
                    {1.2709321,  0.23611228},
                    {1.2667675,  2.6915317},
                    {1.3071702,  -0.27117652},
                    {1.3598092,  -2.8866823},
                    {1.2882001,  -0.69540447},
                    {1.3058296,  -0.431945},
                    {1.9985176,  -1.4783019},
                    {1.3051057,  -1.5100087},
                    {1.2864586,  -0.8126242},
                    {1.0502341,  -2.0869555},
                    {1.3129756,  -0.57569677},
                    {1.4080464,  -1.6524855},
                    {1.4091825,  0.9404328},
                    {1.1260767,  -2.8129675},
                    {1.3875562,  -2.695458},
                    {1.6671637,  2.1589787},
                    {1.0697349,  -1.0930796},
                    {0.98949546,  1.2596153},
                    {1.210892,  -2.6676776},
                    {0.87228537,  1.6205815},
                    {1.2850802,  -1.406912},
                    {1.7240516,  -0.25950402},
                    {1.145118,  -2.7420847},
            };




}
