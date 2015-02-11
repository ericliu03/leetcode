package leetcode;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;



class ListNode {
     int val;
     ListNode next;
     ListNode(int x) {
         val = x;
         next = null;
     }
}

public class Solution {
    public static void main(String[] args) {

    	Solution solution = new Solution();
    	long aa=System.currentTimeMillis();
    	int[][] test = new int[1][1];
    	test[0][0] = 1;
    	int result = solution.uniquePathsWithObstacles(test);//solution.lengthOfLongestSubstring(a);
    	System.out.println("\r<br>执行耗时 : "+(System.currentTimeMillis()-aa)/10000f+" 秒 ");
    	System.out.println(result);
    }   

	public class TreeNode {
		int val;
		TreeNode left;
		TreeNode right;

		TreeNode(int x) {
			val = x;
			left = null;
			right = null;
		}
	}

    public int maxProduct(int[] A) {
        int length = A.length;
        int result = A[0];
        int max = A[0];
        int min = A[0];
        int temp_max;
        for (int i = 1; i < length; i++) {
            temp_max = Math.max(A[i], Math.max(min * A[i], max * A[i]));
            min = Math.min(A[i], Math.min(min * A[i], max * A[i]));
            max = temp_max;
            if (max > result) result = max;
        }
        return result;
    }
    
    public List<TreeNode> generateTrees(int n) {
        ArrayList<TreeNode> trees = new ArrayList<TreeNode>();
        if (n == 0) {
            trees.add(null);
        	return trees;
        }
        TreeNode first = new TreeNode(1);
        trees.add(first);
        for (int i = 2; i <= n; i++) {
        	for (int j = 1; j < i; j++) {
            	TreeNode newNode = new TreeNode(i);
        		trees.get(j - 1).right = newNode;
        	}
        	TreeNode newNode = new TreeNode(i);
        	newNode.left = trees.get(i - 2);
        	trees.add(newNode);
        }
        return trees;
    }
    public int uniquePathsWithObstacles(int[][] obstacleGrid) {
        int m = obstacleGrid.length;
        if (m == 0) return 0;
        int n = obstacleGrid[0].length;
        if (n == 0) return 0;

        int[][] currentPaths = new int[m][n];
        
        
        boolean hasO = false;
        for (int i = 0; i < m; i++) {
            if (obstacleGrid[i][0] == 1) { hasO = true; break; }
            currentPaths[i][0] = 1;
        }
        for (int i = 1; i < n; i++) {
            if (obstacleGrid[0][i] == 1) { hasO = true; break; }
            currentPaths[0][i] = 1;
        }
        
        if (m == 1 || n == 1) {
        	if(hasO) return 0;
        	else return 1;
        }
        
        for(int i = 1; i < m; i++) {
            for (int j = 1; j < n; j++) {
            	currentPaths[i][j] = 0;
            	if (obstacleGrid[i][j] == 1) continue;
            	if (obstacleGrid[i - 1][j] != 1) currentPaths[i][j] += currentPaths[i-1][j];
            	if (obstacleGrid[i - 1][j] != 1) currentPaths[i][j] += currentPaths[i-1][j];
            }
        }
        return currentPaths[m-1][n-1];
    }
    public int maxProfit(int[] prices) {
    	int length = prices.length;
    	if(length < 2) return 0;
        int lowest = prices[0];
        int profit = 0;
        
        for (int i = 1; i < length; i++) {
        	if (prices[i] > lowest) {
        		profit = Math.max(prices[i] - lowest, profit);
        	} else {
        		lowest = prices[i];
        	}
        }
        return Math.max(0, profit);
    }
    public int numDecodings_second(String s) {
        if( s==null || s.length()==0)
            return 0;
        int len = s.length();
        int[] decode = new int[len+1];
        decode[len-1] = (s.charAt(len-1)== '0') ? 0 : 1 ;
        decode[len] =1;
        for(int i = len-2; i>=0; i--) {
            if(s.charAt(i)-'0'==0) {
                decode[i]=0;
            }else if((s.charAt(i)-'0')*10 + s.charAt(i+1)-'0' > 26) {
                decode[i] = decode[i+1];
            } else {
                decode[i] = decode[i+1] + decode[i+2]; 
            }
        }
        return decode[0];
    }
    public int numDecodings_secondrevised(String s) {
        if( s==null || s.length()==0)
            return 0;
        int len = s.length();
        
        int[] decode = new int[len+1];
        
        decode[1] = (s.charAt(0)== '0') ? 0 : 1 ;
        decode[0] =1;
        for(int i = 2; i<=len; i++) {
            if(s.charAt(i-1)-'0'==0) {
                decode[i]=0;
            }else if((s.charAt(i-1)-'0')*10 + s.charAt(i-1)-'0' > 26) {
                decode[i] = decode[i-1];
            } else {
                decode[i] = decode[i-1] + decode[i-2]; 
            }
        }
        return decode[len];
    }
    
    public int numDecodings_first(String s) {
    	int length = s.length();
    	if (length == 0) return 0;
    	if (s.charAt(0) == '0') return 0;
    	else if (length == 1) return 1;
    	
    	int[] result = new int[length];
    	result[0] = 1;
    	int firstTwo = Integer.parseInt(s.substring(0, 2));
    	if ( firstTwo <= 26 && s.charAt(1) != '0') result[1] = 2;
    	else if (firstTwo >= 30 && s.charAt(1) == '0') return 0;
    	else result[1] = 1;

        for (int i = 2; i < length; i++) {
        	result[i] = 0;
        	int singleDigit = Integer.parseInt(s.substring(i, i + 1));
        	int doubleDigit = Integer.parseInt(s.substring(i - 1, i + 1));
        	if (singleDigit == 0) {
        		 if (doubleDigit >= 10 && doubleDigit <= 26) result[i] += result[i-2];
        		 else return 0;
        	} else {
        		result[i] += result[i-1];
        		if (doubleDigit >= 10 && doubleDigit <= 26) result[i] += result[i-2];
        	}

        }
        return result[length - 1];
    }
    
    
    public boolean isInterleave(String s1, String s2, String s3) {
        int l1 = s1.length();
        int l2 = s2.length();
        int l3 = s3.length();
        
        if(l1 + l2 != l3) return false;
        else if(l1 == l3 && s1.equals(s3)) return true;
        else if(l2 == l3 && s2.equals(s3)) return true;
        
        boolean[][] resultMatrix = new boolean[l1 + 1][l2 + 1];
        resultMatrix[0][0] = true;
        for(int i = 1; i <= l1; i++) {
        	if (s1.charAt(i - 1) == s3.charAt(i - 1)) resultMatrix[i][0] = true;
        	else break;
        }
        for(int i = 1; i <= l2; i++) {
        	if (s2.charAt(i - 1) == s3.charAt(i - 1)) resultMatrix[0][i] = true;
        	else break;
        }
        
        for(int i = 1; i <= l1; i++) {
        	for (int j = 1; j <= l2; j++) {
        		if (s1.charAt(i - 1) == s3.charAt(i + j - 1)) resultMatrix[i][j] = resultMatrix[i - 1][j];
        		if (s2.charAt(j - 1) == s3.charAt(i + j - 1)) resultMatrix[i][j] = resultMatrix[i][j - 1] || resultMatrix[i][j];
        	}
        }
        for(int i = 0; i <= l1; i++) {
        	for (int j = 0; j <= l2; j++) {
        		System.out.print(resultMatrix[i][j]);
        		System.out.print('\t');
        	}
        	System.out.print('\n');
        }
        return resultMatrix[l1][l2];
        
        
    }
    public int climbStairs(int n) {
        int[] results = new int[n + 1];
        if (n <= 2) return n;
        results[0] = 0;
        results[1] = 1;
        results[2] = 2;
        for (int i = 3; i <= n; i++) {
        	results[i] = results[i - 1] + results[i - 2];
        }
        return results[n];
    }
    
    public int maxSubArray(int[] A) {
        int[] tempResults = new int[A.length];
        int max = A[0];
        tempResults[0] = A[0];
        for (int i = 1; i < A.length; i++) {
            if (tempResults[i - 1] > 0) tempResults[i] = A[i] + tempResults[i - 1];
            else tempResults[i] = A[i];
        	if (tempResults[i] > max) max = tempResults[i];
        }

        return max;
    }
    public int numTrees(int n) {
        int[] G = new int[n+1];
        G[0] = G[1] = 1;
        for (int i = 2; i <= n; i++) {
            for (int j = 1; j <= i; j++) {
                G[i] += G[j - 1] * G[i - j];
            }
        }
        return G[n];
    }
    
    public int singleNumber(int[] A) {
        HashMap<Integer, Integer> currentNums = new HashMap<Integer, Integer>();
        for(int num: A) {
            if (currentNums.get(num) != null) {
                currentNums.put(num, 1);
            } else {
            	currentNums.put(num, 0);
            }
        }
        for(int num: A) {
        	if (currentNums.get(num) == 0) return num;
        }
        return 0;
    }
    
    public int minimumTotal(List<List<Integer>> triangle) {
    	int numOfRow = triangle.size();
        int[] result = new int[numOfRow];
//        length[0] = triangle.get(0).get(0);
        for (int i = 1; i < numOfRow; i++) {
        	List<Integer> currentRow = triangle.get(i);
        	for (int j = 0; j < currentRow.size(); j++) {
        		int upper = Integer.MIN_VALUE;
        		int leftUpper = Integer.MIN_VALUE;
        		if(j != currentRow.size() - 1) upper = result[j];
        		if(j != 0) leftUpper = result[j - 1];
        		result[j] = Math.max(upper, leftUpper);
        	}
        }
        Arrays.sort(result);
        return result[result.length - 1];
    }
    
    
    public boolean isMatch(String s, String p) {
        if (s==null&&p==null) return true;

        if (s.length()==0&&p.length()==0) return true;

        boolean[][] matrix = new boolean[s.length()+1][p.length()+1];

        matrix[0][0]=true;

        for (int i=1;i<=s.length();i++)
            matrix[i][0]=false;

        for (int j=1;j<=p.length();j++)
            if (p.charAt(j-1)=='*'&&j>1)
                matrix[0][j]=matrix[0][j-2];
            else matrix[0][j]=false;

        for (int i=1;i<=s.length();i++)
            for (int j=1;j<=p.length();j++)

                if (p.charAt(j-1)==s.charAt(i-1)||p.charAt(j-1)=='.')
                    matrix[i][j]=matrix[i-1][j-1];

                else if (p.charAt(j-1)=='*'&&j>1)                   
                    if (p.charAt(j-2)==s.charAt(i-1)||p.charAt(j-2)=='.')
                        matrix[i][j]=matrix[i-1][j]||matrix[i][j-2]||matrix[i][j-1];
                        //matrix[i-1][j]:abb vs ab*: depends on ab vs ab*
                        //matrix[i][j-2] a  vs ab*  depends on a vs a
                        //matrix[i][j-1] ab vs ab*: depends on ab vs ab 
                    else 
                        matrix[i][j]=matrix[i][j-2]; 

                else matrix[i][j]=false;

        return matrix[s.length()][p.length()];
    }
    public boolean isMatch3(String s, String p) {
        int lens = s.length();
        int lenp = p.length();
        if (lens == 0 && lenp == 0) return true;

        boolean[] last = new boolean[lenp + 1]; //last[j] means if p[1~j] matches s[1~i-1]
        boolean[] cur = new boolean[lenp + 1]; //last[j] means if p[1~j] matches s[1~i]
        last[0] = cur[0] = true;
        // for string like "a*b*c*", make last/cur[indexOf('*')]=true.
        for (int j = 1; j <= lenp; j++) {
            if (j >= 2 && p.charAt(j - 1) == '*' && last[j - 2]) {
                last[j] = cur[j] = true;
            }
        }

        for (int i = 1; i <= lens; i++) {
            // determine if p[1~j] matches s[1~i].
            cur[0] = false;
            for (int j = 1; j <= lenp; j++) {
                // three cases: 
                // 1) p[j]==s[i] (include p[j]=='.') and p[1~j-1] matches s[1~i-1];
                // 2) p[j-1~j]="x*" and s[i]='x' and p[1~j] matches s[1~i-1];
                // 3) p[j-2~j]="xy*" and p[1~j-2] matches s[1~i].
                cur[j] = (p.charAt(j - 1) == s.charAt(i - 1) || p.charAt(j - 1) == '.') && last[j - 1]
                        || p.charAt(j - 1) == '*' && (p.charAt(j - 2) == s.charAt(i - 1) || p.charAt(j - 2) == '.') && last[j]
                        || j >= 2 && p.charAt(j - 1) == '*' && cur[j - 2];
            }
            for (int j = 0; j <= lenp; j++) {
                last[j] = cur[j];
            }
        }

        return cur[lenp];
    }
    public boolean isMatch2(String s, String p) {
    	if(s.isEmpty()) return true;
    	int lenS = s.length();
    	int j = 0;
        for(int i = 0; i < p.length(); i++) {
        	char tempChar = p.charAt(i);
        	if(i+1 < p.length() && p.charAt(i+1) == '*') {
        		i++;
        		if(tempChar == '.')
        			while(j < lenS) j++;
        		else
        			while(j < lenS && s.charAt(j) == tempChar) j++;
        	} else if(s.charAt(j) == tempChar) j++;
        	else break;
        	if(j == lenS) return true;
        }
        return false;
    }
    
    public int atoi(String str) {
    	int INT_MAX =2147483647;
        int INT_MIN =-2147483648;
        str = str.trim();
        if(str.isEmpty()) return 0;
        int i = 0;
        if(str.charAt(0)==43||str.charAt(0)==45) i = 1;
        else if(str.charAt(0)>57||str.charAt(0)<48)return 0;

        int j = i;
        while(j < str.length() && str.charAt(j) < 58 && str.charAt(j) > 47) j++;
        if(j == i && j != str.length()) return 0;
        
        double result = 0;
        if(i == 1) {
	        if(str.charAt(0) == '-') {
	        	result = Double.parseDouble('-'+str.substring(1, j));
	        	if(result < INT_MIN) result = INT_MIN;
	        }
	        else if(str.charAt(0) == '+'){
	        	result = Double.parseDouble(str.substring(1, j));
	        } 
        } else {
	    	result = Double.parseDouble(str.substring(0, j));
	    	if(result > INT_MAX) result = INT_MAX;
	    }
        return (int)result;
    }
    

    
    public boolean isPalindrome(int x) {
        if(x<0 || x > 2147483647) return false;
        else if(x<10) return true;
        Integer a = x, b = 1;
        
        while(a!=0) {
        	b = b * 10 ;
        	System.out.println(b);
        			
        			b += a % 10;
        	
        	a = a / 10;
        	
        }
        
        if(b == x)return true;
        else return false;
    }
    public int reverse(int x) {
    	boolean negtive = false;
        StringBuffer b = new StringBuffer(((Integer)x).toString()).reverse();
        
    	if(x < 0) {
    		b.deleteCharAt(b.length()-1);
    		negtive = true;
    	}

        double result = Double.parseDouble(b.toString());
        if(result > 2147483648L) return 0;
        if(negtive) result = -result;
        return (int)result;
    }
    
    public boolean isPalindrome(String s, int startIndex, int endIndex) {
    	for(int i = startIndex, j = endIndex; i <= j; i++, j--) 
    			if (s.charAt(i) != s.charAt(j)) return false;
    	return true;
    }
    
    public String longestPalindrome(String s) {
    	int n = s.length();
    	int longestLen = 0;
    	int longestIndex = 0;
    	
    	for(int currentIndex = 0; currentIndex < n; currentIndex++) {
    		if(isPalindrome(s,currentIndex - longestLen, currentIndex)){
    			longestLen += 1;
    			longestIndex = currentIndex;
    		} else if(currentIndex - longestLen - 1 >= 0 && 
    				  isPalindrome(s, currentIndex - longestLen - 1, currentIndex)) {
    			longestLen += 2;
    			longestIndex = currentIndex;
    		}	
    	}
    	longestIndex++;
    	return s.substring(longestIndex - longestLen, longestIndex);
    }
    
    public String longestPalindromePre(String s) {
    	//return a new string start with ^ and end with $, and add #s: #a#b#c#.
    	int n = s.length();
    	if (n == 0) return "^$";
    	String ret = "^";
    	for(int i = 0; i < n; i++) 
    		ret+= "#" + s.substring(i, i + 1);

    	ret += "#$";
    	return ret;
    }
    public String longestPalindrome_second(String s) {
    	String T = longestPalindromePre(s);
    	
    	int n = T.length();
    	int[] P = new int[n];//Point
    	int C = 0, R = 0; //Center, Range
    	for (int i = 1; i < n - 1; i++) {
    		int i_mirror = 2*C -i;
    		P[i] = (R > i)?Math.min(R - i, P[i_mirror]):0;
    		
    		while(T.charAt(i + P[i] + 1) == T.charAt(i - P[i] - 1)) P[i]++;
    		
    		if(i + P[i] > R) {
    			C = i;
    			R = i + P[i];
    		}
    	}
    	
    	int maxLen = 0;
    	int centerIndex = 0;
    	for (int i = 1; i < n - 1; i++) {
    		if (P[i]>maxLen) {
    			maxLen = P[i];
    			centerIndex = i;
    		}
    	}
    	int startP = (centerIndex - maxLen - 1)/2;
    	return s.substring(startP, startP + maxLen);
    }
    
    public String longestPalindrome_first(String s) {
    	
    	if(s.length()==0) return null;
    	else if(s.length() == 1) return s;

    	String longestStr = new String();
    	StringBuilder tempStr = new StringBuilder();
    	int length = s.length();
    	
    	int Loop = 1;
    	
    	for(int i = 0; i < length - 1; i++) {
    		int halfLong = longestStr.length()/2;
    		
    		tempStr = new StringBuilder(s.substring(i, i+1));
    		int former = i, latter = i;
    		
    		if(s.charAt(i) == s.charAt(i+1) && Loop == 1) {
    			latter++;
    			tempStr.append(s.charAt(i));
    			Loop++;
    			i--;
    		} else if(Loop == 2) Loop = 1;
    		
    		if(former < halfLong )continue;
    		else if(length - latter < halfLong) break;



    		String formerStr = new StringBuffer(s.substring(former - halfLong, former)).reverse().toString();
    		if(formerStr == s.substring(latter, latter + halfLong)) {
    			former -= halfLong;
    			latter += halfLong;
    		}
    		
    		while(true) {
    			former--;
    			latter++;
    			if(former < 0 || latter > length - 1) break;
    			if(s.charAt(former) != s.charAt(latter)) break;
    			tempStr.insert(0, s.charAt(former));
    			tempStr.append(s.charAt(latter));
    		}
    		if(tempStr.length() > longestStr.length()) longestStr = tempStr.toString();
    		//if(s.charAt(i) == s.charAt(i+1)) {}
    	}
    	return longestStr;
    }

    
    public double findMedianSortedArrays(int A[], int B[]) {
        int lengthA = A.length;
        int lengthB = B.length;
        int total = lengthA + lengthB;

        int iMin = 0;
        int iMax = lengthA;
        while(iMin <= iMax) {

        	int i = (iMin + iMax) / 2;
        	int j = (lengthA + lengthB + 1) / 2 - i; //如果是奇数多加1 保证占一半以上，这样取数的时候取a和b里大的那个
        	
        	if(A[i] < B[j - 1] && i < lengthA && j > 0) iMin = i + 1;
        	else if(A[i - 1] > B[j] && i > 0 && j < lengthB) iMax = i - 1;
        	else {
        		System.out.println(i);
        		
        		System.out.println(j);
        		
        		
        		if(total%2 == 1){
        			if(i == 0) return B[j-1];
        			return Math.max(A[i-1], B[j-1]);
        		} else {
        			return (A[i-1] + B[j-1]) / 2.0 ;
        			
        		}
        	
        	}
        	System.out.println(iMin);
        	System.out.println(iMax);
        }
    	
    	
    	
		return 13211;
    	
    }
    public double findKth(int A[], int B[], int lengthA, int lengthB, int k) {
    	if (lengthA > lengthB) return findKth(B, A, lengthB, lengthA, k);
    	if (lengthA == 0) return B[k - 1];
    	if (k == 1) return Math.min(A[0], B[0]);
    	
    	int pointerA = Math.min(lengthA, k/2);
    	int pointerB = k - pointerA;
    	if(A[pointerA - 1] < B[pointerB - 1]) return findKth(Arrays.copyOfRange(A, pointerA, lengthA), B, lengthA - pointerA, lengthB, k - pointerA);
    	else if (A[pointerA - 1] > B[pointerB - 1]) return findKth(A, Arrays.copyOfRange(B, pointerB, lengthB), lengthA, lengthB - pointerB, k - pointerA);
    	else return A[pointerA - 1];
    }
    
    public double findMedianSortedArrays_first(int A[], int B[]) {
        int lengthA = A.length;
        int lengthB = B.length;
        int total = lengthA + lengthB;
        
        if (total%2 == 1)
        	return findKth(A, B, lengthA, lengthB, total/2 + 1);
        else {
        	return ( findKth(A, B, lengthA, lengthB, total/2) + findKth(A, B, lengthA, lengthB, total/2 + 1)) / 2;
        }
    }
    
    
    String convert2(String s, int nRows) {
        // special case: nRows == 1
        if (nRows == 1) return s;

        String ss = "";
        int len = s.length();

        // deal with the first line
        for(int i = 0; i < len; i += ((nRows-1) * 2))
            ss += s.charAt(i);

        // deal the rest
        for(int line = 2; line <= nRows-1; line++) {
            // add two chars at once
            for(int i = line-1; i < len; i += ((line - 1) * 2)) {
                // add char one
                if (i < len) ss += s.charAt(i);
                // move to next
                i += ((nRows - line) * 2);
                // add char two
                if (i < len) ss += s.charAt(i);
            }
        }

        // deal with the last line, same as the first line
        for(int i = nRows-1; i < len; i += ((nRows-1) * 2))
            ss += s.charAt(i);

        return ss;
    }
    
    public String convert(String s, int nRows) {
        if(s == null) return null;
        else if(s.length() == 1 || nRows == 1) return s;
        
        StringBuilder[] stringRows = new StringBuilder[nRows];
        for(int i = 0; i < nRows; i++) stringRows[i] = new StringBuilder();
        int row = 0;
        boolean downward = true;
        
        for(int index = 0; index < s.length(); index ++) {
        	stringRows[row].append(s.charAt(index));
        	
        	if(row == 0) downward = true;
        	else if(row == (nRows - 1)) downward = false;
        	
        	if(downward) row++;
        	else row--;
        }
        
        StringBuilder result = new StringBuilder();
        for(StringBuilder string: stringRows) {
        	result.append(string.toString());
        }
        
        return result.toString();
    }

	public int lengthOfLongestSubstring_third(String s) {
		//must be letters that in ASCII first 128
	    int array[]=new int[128];
	    Arrays.fill(array,-1);
	    int len=0;
	    int start=-1;
	    for(int i=0;i<s.length();i++) {
	    	//if a letter appears whose last position is after the start point, move the start to there
            start=start>=array[s.charAt(i)]?start:array[s.charAt(i)];
            //no matter the start is moved or not, calculate the longest line( a little waste here)
            len=(i-start)>len?(i-start):len; // calculated before move to next position, so the start point starts at -1 and move directly to array's i
            
            array[s.charAt(i)]=i;
	    }
	    return len;
	}
	public int lengthOfLongestSubstring(String s) {
		int length = s.length();
		if (length == 0) return 0;
		
		Integer startPoint = 0;
		Integer maxLength = 0;
		
		HashMap<Character, Integer> letterSeens = new HashMap<Character, Integer>();
		
		for (int i = 0; i < length; i++) {
			Integer lastSeen = letterSeens.get(s.charAt(i));
			if(lastSeen != null && lastSeen >= startPoint) {
				//from the point lastSeen to current point i
				//there is no possible substring since these two letters are same
				maxLength = Math.max(maxLength, i - startPoint);
				startPoint = lastSeen + 1;
			}
			
			letterSeens.put(s.charAt(i), i);
		}
		maxLength = Math.max(maxLength, length - startPoint);
		return maxLength;
	
	}
    public int lengthOfLongestSubstring_first(String s) {
    	int subStringLength = 0;
    	HashMap<Character, Integer> hashTable = new HashMap<Character, Integer>();
    	HashMap<Character, Boolean> dupTable = new HashMap<Character, Boolean>();
    	int i = 0, j = 0;
    	char tempLetter;
    	int tempInt;
        while(j < s.length()) {

        	//if no duplicated letter in the window, add next letters

        	if(!dupTable.containsValue(true)) {	
        		while(j < s.length() && (!hashTable.containsKey(s.charAt(j)) || (hashTable.get(s.charAt(j)) == 0))){
		    		hashTable.put(s.charAt(j), 1);
		    		subStringLength++;
		    		j++;
		    	}
            if (j >= s.length()) break;
        	}
        	


        	//deal with letter that is removed

        	tempLetter = s.charAt(i);
        	tempInt = hashTable.get(s.charAt(i)) - 1;
        	hashTable.put(tempLetter, tempInt);
        	if(tempInt <= 1) dupTable.put(tempLetter, false);
        	i++;
        	
        	//deal with letter that is added 
        	tempLetter = s.charAt(j);
        	if (hashTable.containsKey(tempLetter)) {
        		tempInt =  1 + hashTable.get(tempLetter); 
        		if(tempInt > 1) dupTable.put(tempLetter, true);
        	}
        	else tempInt = 1;
        	hashTable.put(tempLetter, tempInt);
        	
        	j++;
        }
        return subStringLength;
    }
    

}