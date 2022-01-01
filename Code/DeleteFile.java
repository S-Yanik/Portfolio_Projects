
// Java program to delete a file  
import java.io.*;

public class DeleteFile
{
    public static void main(String[] args)
    {
        File file = new File("C:\\Users\\Sean\\Desktop\\0.7394271327934021");

        if(file.delete())
        {
            System.out.println("File deleted successfully");
        }
        else
        {
            System.out.println("Failed to delete the file");
        }
    }
} 