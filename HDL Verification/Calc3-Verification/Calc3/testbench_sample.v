// Library:  calc3
// Module:  Testbenches
// Author:  Naseer Siddique

`include "calc3_top.v"

// the test below tests a simple store, add, and fetch operation

module atestsimple;
   
   
   wire [0:1] out1_resp, out1_tag, out2_resp, out2_tag, out3_resp, out3_tag, out4_resp, out4_tag;
   wire [0:31] out1_data, out2_data, out3_data, out4_data;
   wire 	 scan_out;
   
   reg [0:1] req1_tag, req2_tag, req3_tag, req4_tag;
   reg       a_clk, b_clk, c_clk, reset, scan_in;
   reg [0:3] req1_cmd, req1_d1, req1_d2, req1_r1, req2_cmd, req2_d1, req2_d2, req2_r1, req3_cmd, req3_d1, req3_d2, req3_r1, req4_cmd, req4_d1, req4_d2, req4_r1;
   reg [0:31] req1_data, req2_data, req3_data, req4_data;
      
   wire [0:15] add_shift_branch_data;
   wire [0:3]  adder_cmd, adder_out_cmd;
   wire [0:4]  adder_follow_branch;
   wire [0:31] adder_out_data1, adder_out_data2, adder_out_data3, adder_out_data4;
   wire [0:1]  adder_out_resp1, adder_out_resp2, adder_out_resp3, adder_out_resp4, adder_out_tag1, adder_out_tag2, adder_out_tag3, adder_out_tag4;
   wire        adder_overflow;
   wire [0:3]  adder_read_adr1, adder_read_adr2; 
   wire [0:63] adder_read_d1, adder_read_d2; 
   wire        adder_read_valid1, adder_read_valid2;
   wire [0:63] adder_result;
   wire [0:4]  adder_result_reg;
   wire [0:3]  adder_tag, adder_write_adr;
   wire [0:31] adder_write_data;
   wire        adder_write_valid;
   wire [0:3]  hold1_cmd, hold1_d1, hold1_d2;
   wire [0:31] hold1_data;
   wire [0:3]  hold1_r1;
   wire [0:1]  hold1_tag;
   wire [0:3]  hold2_cmd, hold2_d1, hold2_d2;
   wire [0:31] hold2_data;
   wire [0:3]  hold2_r1;
   wire [0:1]  hold2_tag;
   wire [0:3]  hold3_cmd, hold3_d1, hold3_d2;
   wire [0:31] hold3_data;
   wire [0:3]  hold3_r1;
   wire [0:1]  hold3_tag;
   wire [0:3]  hold4_cmd;
   wire [0:3]  hold4_d1,  hold4_d2;
   wire [0:31] hold4_data;
   wire [0:3]  hold4_r1;
   wire [0:1]  hold4_tag;
   wire port1_invalid_op;
   wire [0:1] port1_invalid_tag;
   wire       port2_invalid_op;
   wire [0:1] port2_invalid_tag;
   wire       port3_invalid_op;
   wire [0:1] port3_invalid_tag;
   wire       port4_invalid_op;
   wire [0:1] port4_invalid_tag;
   wire [0:3] prio_adder_cmd;
   wire [0:4] prio_adder_data1, prio_adder_data2;
   wire [0:4] prio_adder_follow_branch;
   wire       prio_adder_out_vld;
   wire [0:4] prio_adder_result;
   wire [0:3] prio_adder_tag,  prio_shift_cmd;
   wire [0:31] prio_shift_data;
   wire [0:4]  prio_shift_data1,  prio_shift_data2;
   wire [0:4]  prio_shift_follow_branch;
   wire        prio_shift_out_vld;
   wire [0:4]  prio_shift_result;
   wire [0:3]  prio_shift_tag;
   wire        scan_connect1, scan_connect10, scan_connect11, scan_connect12, scan_connect2, scan_connect3, scan_connect4, scan_connect5, scan_connect6, scan_connect7, scan_connect8, scan_connect9;
   wire [0:3]  shift_cmd;
   wire [0:4]  shift_follow_branch;
   wire [0:3]  shift_out_cmd;
   wire [0:31] shift_out_data1, shift_out_data2, shift_out_data3, shift_out_data4;
   wire [0:1]  shift_out_resp1;
   wire [0:1]  shift_out_resp2, shift_out_resp3, shift_out_resp4, shift_out_tag1, shift_out_tag2, shift_out_tag3, shift_out_tag4;
   wire [0:3]  shift_read_adr1,  shift_read_adr2;
   wire [0:63] shift_read_d1, shift_read_d2;
   wire        shift_read_valid1,  shift_read_valid2;
   wire [0:63] shift_result;
   wire [0:4]  shift_result_reg;
   wire [0:3]  shift_tag, shift_write_adr;
   wire [0:31] shift_write_data;
   wire        shift_write_valid, store_data_valid;
   wire [0:63] store_val;
   
   calc3_top ( out1_data, out1_resp, out1_tag, out2_data, out2_resp, out2_tag, out3_data, out3_resp, out3_tag, out4_data, out4_resp, out4_tag, scan_out, a_clk, b_clk, c_clk, req1_cmd, req1_d1, req1_d2, req1_data, req1_r1, req1_tag, req2_cmd, req2_d1, req2_d2, req2_data, req2_r1, req2_tag, req3_cmd, req3_d1, req3_d2, req3_data, req3_r1, req3_tag, req4_cmd, req4_d1, req4_d2, req4_data, req4_r1, req4_tag, reset, scan_in);
      
   initial 
     begin
	c_clk = 0;
	a_clk = 0;
	b_clk = 0;
	scan_in = 0;
	
     end
   
   
   always #100 c_clk = ~c_clk;
   
   initial
     begin
	
	reset = 1;
	
	
	
	#600 

	  reset = 0;
	req1_cmd = 0;
	req1_d1 = 0;
	req1_d2 = 0;
	req1_data = 0 ;
	req1_r1 = 0;
	req1_tag = 0;
	req2_cmd = 0;
	req2_d1 = 0;
	req2_d2 = 0;
	req2_data = 0 ;
	req2_r1 = 0;
	req2_tag = 0;
	req3_cmd = 0;
	req3_d1 = 0;
	req3_d2 = 0;
	req3_data = 0 ;
	req3_r1 = 0;
	req3_tag = 0;
	req4_cmd = 0;
	req4_d1 = 0;
	req4_d2 = 0;
	req4_data = 0 ;
	req4_r1 = 0;
	req4_tag = 0;
	
	#200 
	  
	  req1_cmd = 9;
	req1_d1 = 0;
	req1_d2 = 0;
	req1_data = 'b101010101010101010 ;
	req1_r1 = 4;
	req1_tag = 1;
	
	#200 

	req1_cmd = 0;
	req1_d1 = 0;
	req1_d2 = 0;
	req1_data = 0 ;
	req1_r1 = 0;
	req1_tag = 0;
	
	#200

	  req1_cmd = 9;
	req1_d1 = 0;
	req1_d2 = 0;
	req1_data = 'b1111100111110011111 ;
	req1_r1 = 5;
	req1_tag = 2;
		
	#200

	req1_cmd = 0;
	req1_d1 = 0;
	req1_d2 = 0;
	req1_data = 0 ;
	req1_r1 = 0;
	req1_tag = 0;

	#600
	  
	  req1_cmd = 1;
	req1_d1 = 4;
	req1_d2 = 5;
	req1_data = 0 ;
	req1_r1 = 6;
	req1_tag = 3;

	#200

	req1_cmd = 0;
	req1_d1 = 0;
	req1_d2 = 0;
	req1_data = 0 ;
	req1_r1 = 0;
	req1_tag = 0;

	#600

	  req1_cmd = 10;
	req1_d1 = 6;
	req1_d2 = 0;
	req1_data = 0 ;
	req1_r1 = 0;
	req1_tag = 0;

	#200
	  
	  req1_cmd = 0;
	req1_d1 = 0;
	req1_d2 = 0;
	req1_data = 0 ;
	req1_r1 = 0;
	req1_tag = 0;
	
	#2000 $stop;

     end // initial begin
   
   
   always
     @ (reset or c_clk or req1_cmd or req1_data or req2_cmd or req2_data or req3_cmd or req3_data or req4_cmd or req4_data) begin
	
	$display ("%t: r:%b \n 
1c:%d,1d1:%d,1d2:%d,1d:%d,1r:%d,1t:%d
2c:%d,2d1:%d,2d2:%d,2d:%d,2r:%d,2t:%d
3c:%d,3d1:%d,3d2:%d,3d:%d,3r:%d,3t:%d
4c:%d,4d1:%d,4d2:%d,4d:%d,4r:%d,4t:%d  
1r:%d,1d:%d,1t:%d  
2r:%d,2d:%d,2t:%d  
3r:%d,3d:%d,3t:%d  
4r:%d,4d:%d,4t:%d \n\n", 
		       
		  $time, reset, 
		  req1_cmd, req1_d1, req1_d2, req1_data, req1_r1, req1_tag,
		  req2_cmd, req2_d1, req2_d2, req2_data, req2_r1, req2_tag,
		  req3_cmd, req3_d1, req3_d2, req3_data, req3_r1, req3_tag,
		  req4_cmd, req4_d1, req4_d2, req4_data, req4_r1, req4_tag,
		  out1_resp, out1_data, out1_tag,
		  out2_resp, out2_data, out2_tag,
		  out3_resp, out3_data, out3_tag,
		  out4_resp, out4_data, out4_tag);
	
     end // always @ (reset or c_clk or req1_cmd or req1_data or req2_cmd or req2_data or req3_cmd or req3_data or req4_cmd or req4_data)
   
   
endmodule // testsimple

module atestreqs;

   wire [0:1] out1_resp, out1_tag, out2_resp, out2_tag, out3_resp, out3_tag, out4_resp, out4_tag;
   wire [0:31] out1_data, out2_data, out3_data, out4_data;
   wire 	 scan_out;
   
   reg [0:1] req1_tag, req2_tag, req3_tag, req4_tag;
   reg       a_clk, b_clk, c_clk, reset, scan_in;
   reg [0:3] req1_cmd, req1_d1, req1_d2, req1_r1, req2_cmd, req2_d1, req2_d2, req2_r1, req3_cmd, req3_d1, req3_d2, req3_r1, req4_cmd, req4_d1, req4_d2, req4_r1;
   reg [0:31] req1_data, req2_data, req3_data, req4_data;
      
   wire [0:15] add_shift_branch_data;
   wire [0:3]  adder_cmd, adder_out_cmd;
   wire [0:4]  adder_follow_branch;
   wire [0:31] adder_out_data1, adder_out_data2, adder_out_data3, adder_out_data4;
   wire [0:1]  adder_out_resp1, adder_out_resp2, adder_out_resp3, adder_out_resp4, adder_out_tag1, adder_out_tag2, adder_out_tag3, adder_out_tag4;
   wire        adder_overflow;
   wire [0:3]  adder_read_adr1, adder_read_adr2; 
   wire [0:63] adder_read_d1, adder_read_d2; 
   wire        adder_read_valid1, adder_read_valid2;
   wire [0:63] adder_result;
   wire [0:4]  adder_result_reg;
   wire [0:3]  adder_tag, adder_write_adr;
   wire [0:31] adder_write_data;
   wire        adder_write_valid;
   wire [0:3]  hold1_cmd, hold1_d1, hold1_d2;
   wire [0:31] hold1_data;
   wire [0:3]  hold1_r1;
   wire [0:1]  hold1_tag;
   wire [0:3]  hold2_cmd, hold2_d1, hold2_d2;
   wire [0:31] hold2_data;
   wire [0:3]  hold2_r1;
   wire [0:1]  hold2_tag;
   wire [0:3]  hold3_cmd, hold3_d1, hold3_d2;
   wire [0:31] hold3_data;
   wire [0:3]  hold3_r1;
   wire [0:1]  hold3_tag;
   wire [0:3]  hold4_cmd;
   wire [0:3]  hold4_d1,  hold4_d2;
   wire [0:31] hold4_data;
   wire [0:3]  hold4_r1;
   wire [0:1]  hold4_tag;
   wire port1_invalid_op;
   wire [0:1] port1_invalid_tag;
   wire       port2_invalid_op;
   wire [0:1] port2_invalid_tag;
   wire       port3_invalid_op;
   wire [0:1] port3_invalid_tag;
   wire       port4_invalid_op;
   wire [0:1] port4_invalid_tag;
   wire [0:3] prio_adder_cmd;
   wire [0:4] prio_adder_data1, prio_adder_data2;
   wire [0:4] prio_adder_follow_branch;
   wire       prio_adder_out_vld;
   wire [0:4] prio_adder_result;
   wire [0:3] prio_adder_tag,  prio_shift_cmd;
   wire [0:31] prio_shift_data;
   wire [0:4]  prio_shift_data1,  prio_shift_data2;
   wire [0:4]  prio_shift_follow_branch;
   wire        prio_shift_out_vld;
   wire [0:4]  prio_shift_result;
   wire [0:3]  prio_shift_tag;
   wire        scan_connect1, scan_connect10, scan_connect11, scan_connect12, scan_connect2, scan_connect3, scan_connect4, scan_connect5, scan_connect6, scan_connect7, scan_connect8, scan_connect9;
   wire [0:3]  shift_cmd;
   wire [0:4]  shift_follow_branch;
   wire [0:3]  shift_out_cmd;
   wire [0:31] shift_out_data1, shift_out_data2, shift_out_data3, shift_out_data4;
   wire [0:1]  shift_out_resp1;
   wire [0:1]  shift_out_resp2, shift_out_resp3, shift_out_resp4, shift_out_tag1, shift_out_tag2, shift_out_tag3, shift_out_tag4;
   wire [0:3]  shift_read_adr1,  shift_read_adr2;
   wire [0:63] shift_read_d1, shift_read_d2;
   wire        shift_read_valid1,  shift_read_valid2;
   wire [0:63] shift_result;
   wire [0:4]  shift_result_reg;
   wire [0:3]  shift_tag, shift_write_adr;
   wire [0:31] shift_write_data;
   wire        shift_write_valid, store_data_valid;
   wire [0:63] store_val;
   
   calc3_top ( out1_data, out1_resp, out1_tag, out2_data, out2_resp, out2_tag, out3_data, out3_resp, out3_tag, out4_data, out4_resp, out4_tag, scan_out, a_clk, b_clk, c_clk, req1_cmd, req1_d1, req1_d2, req1_data, req1_r1, req1_tag, req2_cmd, req2_d1, req2_d2, req2_data, req2_r1, req2_tag, req3_cmd, req3_d1, req3_d2, req3_data, req3_r1, req3_tag, req4_cmd, req4_d1, req4_d2, req4_data, req4_r1, req4_tag, reset, scan_in);
      
   initial 
     begin
	c_clk = 0;
	a_clk = 0;
	b_clk = 0;
	scan_in = 0;
	reset = 1;
		
     end
   
	
   always #100 c_clk = ~c_clk;
   
   initial
     begin

	#600 

	  reset = 0;
	req1_cmd = 0;
	req1_d1 = 0;
	req1_d2 = 0;
	req1_data = 0 ;
	req1_r1 = 0;
	req1_tag = 0;
	req2_cmd = 0;
	req2_d1 = 0;
	req2_d2 = 0;
	req2_data = 0 ;
	req2_r1 = 0;
	req2_tag = 0;
	req3_cmd = 0;
	req3_d1 = 0;
	req3_d2 = 0;
	req3_data = 0 ;
	req3_r1 = 0;
	req3_tag = 0;
	req4_cmd = 0;
	req4_d1 = 0;
	req4_d2 = 0;
	req4_data = 0 ;
	req4_r1 = 0;
	req4_tag = 0;
	
	#200 

	  // TEST REQUESTOR 1 INDIVIDUALLY
	  
	  // store 5 -> reg 1

	  req1_cmd = 9;
	req1_d1 = 0;
	req1_d2 = 0;
	req1_data = 5 ;
	req1_r1 = 1;
	req1_tag = 0;

	#200

	req1_cmd = 0;
	req1_d1 = 0;
	req1_d2 = 0;
	req1_data = 0 ;
	req1_r1 = 0;
	req1_tag = 0;

	#200 

	  // store 10 -> reg 2
	  
	req1_cmd = 9;
	req1_d1 = 0;
	req1_d2 = 0;
	req1_data = 10 ;
	req1_r1 = 2;
	req1_tag = 1;
	
	#200

	req1_cmd = 0;
	req1_d1 = 0;
	req1_d2 = 0;
	req1_data = 0 ;
	req1_r1 = 0;
	req1_tag = 0;

	#200

	  // reg 1 (5) + reg 2 (10) -> reg 4
	  
	req1_cmd = 1;
	req1_d1 = 1;
	req1_d2 = 2;
	req1_data = 0 ;
	req1_r1 = 4;
	req1_tag = 2;
		
	#200

	req1_cmd = 0;
	req1_d1 = 0;
	req1_d2 = 0;
	req1_data = 0 ;
	req1_r1 = 0;
	req1_tag = 0;

	#200

	  // fetch reg 4 (15)
	  
	req1_cmd = 10;
	req1_d1 = 4;
	req1_d2 = 0;
	req1_data = 0 ;
	req1_r1 = 0;
	req1_tag = 3;
			
	#200

	  req1_cmd = 0;
	req1_d1 = 0;
	req1_d2 = 0;
	req1_data = 0 ;
	req1_r1 = 0;
	req1_tag = 0;

	#200

	  // reg 2 (10) - reg 1 (5) -> reg 5
	  
	  req1_cmd = 2;
	req1_d1 = 2;
	req1_d2 = 1;
	req1_data = 0 ;
	req1_r1 = 5;
	req1_tag = 0;

	#200

	req1_cmd = 0;
	req1_d1 = 0;
	req1_d2 = 0;
	req1_data = 0 ;
	req1_r1 = 0;
	req1_tag = 0;

	#200

	  // fetch reg 5
		  
	req1_cmd = 10;
	req1_d1 = 5;
	req1_d2 = 0;
	req1_data = 0 ;
	req1_r1 = 4;
	req1_tag = 1;

	#200

	req1_cmd = 0;
	req1_d1 = 0;
	req1_d2 = 0;
	req1_data = 0 ;
	req1_r1 = 0;
	req1_tag = 0;
	
	#200

	  // reg 2 ('b1010) << reg 1 (5) -> reg 6
		  
	  req1_cmd = 5;
	req1_d1 = 2;
	req1_d2 = 1;
	req1_data = 0 ;
	req1_r1 = 6;
	req1_tag = 2;

		#200

	req1_cmd = 0;
	req1_d1 = 0;
	req1_d2 = 0;
	req1_data = 0 ;
	req1_r1 = 0;
	req1_tag = 0;
	
	#200

	  // fetch reg 6 ('b101000000)
	  
	req1_cmd = 10;
	req1_d1 = 6;
	req1_d2 = 0;
	req1_data = 0 ;
	req1_r1 = 0;
	req1_tag = 3;

	#200

	req1_cmd = 0;
	req1_d1 = 0;
	req1_d2 = 0;
	req1_data = 0 ;
	req1_r1 = 0;
	req1_tag = 0;

	#200

	  // reg 2 (10) >> reg 1 (5) -> reg 7
	  
	  req1_cmd = 6;
	req1_d1 = 2;
	req1_d2 = 1;
	req1_data = 0 ;
	req1_r1 = 7;
	req1_tag = 0;

	#200

	req1_cmd = 0;
	req1_d1 = 0;
	req1_d2 = 0;
	req1_data = 0 ;
	req1_r1 = 0;
	req1_tag = 0;
		
	#200

	  // fetch reg 7 ( 0 )
	  
	req1_cmd = 10;
	req1_d1 = 7;
	req1_d2 = 0;
	req1_data = 0 ;
	req1_r1 = 0;
	req1_tag = 1;
		
	#200

	req1_cmd = 0;
	req1_d1 = 0;
	req1_d2 = 0;
	req1_data = 0 ;
	req1_r1 = 0;
	req1_tag = 0;

	#200

	  // branch if reg 8 == 0
	  
	  req1_cmd = 12;
	req1_d1 = 8;
	req1_d2 = 0;
	req1_data = 0;
	req1_r1 = 0;
	req1_tag = 2;

	#200

	req1_cmd = 0;
	req1_d1 = 0;
	req1_d2 = 0;
	req1_data = 0 ;
	req1_r1 = 0;
	req1_tag = 0;

	#200	  

	  // reg 2 + reg 3 (should be skipped due to branch)

	  req1_cmd = 1;
	req1_d1 = 2;
	req1_d2 = 3;
	req1_data = 0;
	req1_r1 = 9;
	req1_tag = 3;

	#200
	  
	  req1_cmd = 0;
	req1_d1 = 0;
	req1_d2 = 0;
	req1_data = 0 ;
	req1_r1 = 0;
	req1_tag = 0;

	#200

	  // branch if reg 1 == reg5
	  
	  req1_cmd = 13;
	req1_d1 = 1;
	req1_d2 = 5;
	req1_data = 0;
	req1_r1 = 0;
	req1_tag = 0;

	#200

	req1_cmd = 0;
	req1_d1 = 0;
	req1_d2 = 0;
	req1_data = 0 ;
	req1_r1 = 0;
	req1_tag = 0;

	#200	  

	  // reg 2 + reg 3 (should be skipped due to branch)

	  req1_cmd = 1;
	req1_d1 = 2;
	req1_d2 = 3;
	req1_data = 0;
	req1_r1 = 11;
	req1_tag = 1;

	#200
	  
	  req1_cmd = 0;
	req1_d1 = 0;
	req1_d2 = 0;
	req1_data = 0 ;
	req1_r1 = 0;
	req1_tag = 0;


	// TEST REQUESTOR 2 INDIVIDUALLY
	
	#2000 

	  reset = 1;

	#600 reset = 0;
	
	  // store 5 -> reg 1

	  req2_cmd = 9;
	req2_d1 = 0;
	req2_d2 = 0;
	req2_data = 5 ;
	req2_r1 = 1;
	req2_tag = 0;

	#200

	req2_cmd = 0;
	req2_d1 = 0;
	req2_d2 = 0;
	req2_data = 0 ;
	req2_r1 = 0;
	req2_tag = 0;

	#200 

	  // store 10 -> reg 2
	  
	req2_cmd = 9;
	req2_d1 = 0;
	req2_d2 = 0;
	req2_data = 10 ;
	req2_r1 = 2;
	req2_tag = 1;
	
	#200

	req2_cmd = 0;
	req2_d1 = 0;
	req2_d2 = 0;
	req2_data = 0 ;
	req2_r1 = 0;
	req2_tag = 0;

	#200

	  // reg 1 (5) + reg 2 (10) -> reg 4
	  
	req2_cmd = 1;
	req2_d1 = 1;
	req2_d2 = 2;
	req2_data = 0 ;
	req2_r1 = 4;
	req2_tag = 2;
		
	#200

	req2_cmd = 0;
	req2_d1 = 0;
	req2_d2 = 0;
	req2_data = 0 ;
	req2_r1 = 0;
	req2_tag = 0;

	#200

	  // fetch reg 4 (15)
	  
	req2_cmd = 10;
	req2_d1 = 4;
	req2_d2 = 0;
	req2_data = 0 ;
	req2_r1 = 0;
	req2_tag = 3;
			
	#200

	  req2_cmd = 0;
	req2_d1 = 0;
	req2_d2 = 0;
	req2_data = 0 ;
	req2_r1 = 0;
	req2_tag = 0;

	#200

	  // reg 2 (10) - reg 1 (5) -> reg 5
	  
	  req2_cmd = 2;
	req2_d1 = 2;
	req2_d2 = 1;
	req2_data = 0 ;
	req2_r1 = 5;
	req2_tag = 0;

	#200

	req2_cmd = 0;
	req2_d1 = 0;
	req2_d2 = 0;
	req2_data = 0 ;
	req2_r1 = 0;
	req2_tag = 0;

	#200

	  // fetch reg 5
		  
	req2_cmd = 10;
	req2_d1 = 5;
	req2_d2 = 0;
	req2_data = 0 ;
	req2_r1 = 4;
	req2_tag = 1;

	#200

	req2_cmd = 0;
	req2_d1 = 0;
	req2_d2 = 0;
	req2_data = 0 ;
	req2_r1 = 0;
	req2_tag = 0;
	
	#200

	  // reg 2 ('b1010) << reg 1 (5) -> reg 6
		  
	  req2_cmd = 5;
	req2_d1 = 2;
	req2_d2 = 1;
	req2_data = 0 ;
	req2_r1 = 6;
	req2_tag = 2;

		#200

	req2_cmd = 0;
	req2_d1 = 0;
	req2_d2 = 0;
	req2_data = 0 ;
	req2_r1 = 0;
	req2_tag = 0;
	
	#200

	  // fetch reg 6 ('b101000000)
	  
	req2_cmd = 10;
	req2_d1 = 6;
	req2_d2 = 0;
	req2_data = 0 ;
	req2_r1 = 0;
	req2_tag = 3;

	#200

	req2_cmd = 0;
	req2_d1 = 0;
	req2_d2 = 0;
	req2_data = 0 ;
	req2_r1 = 0;
	req2_tag = 0;

	#200

	  // reg 2 (10) >> reg 1 (5) -> reg 7
	  
	  req2_cmd = 6;
	req2_d1 = 2;
	req2_d2 = 1;
	req2_data = 0 ;
	req2_r1 = 7;
	req2_tag = 0;

	#200

	req2_cmd = 0;
	req2_d1 = 0;
	req2_d2 = 0;
	req2_data = 0 ;
	req2_r1 = 0;
	req2_tag = 0;
		
	#200

	  // fetch reg 7 ( 0 )
	  
	req2_cmd = 10;
	req2_d1 = 7;
	req2_d2 = 0;
	req2_data = 0 ;
	req2_r1 = 0;
	req2_tag = 1;
		
	#200

	req2_cmd = 0;
	req2_d1 = 0;
	req2_d2 = 0;
	req2_data = 0 ;
	req2_r1 = 0;
	req2_tag = 0;

	#1000

	  reset = 1;
	
	// TEST REQUESTOR 3 INDIVIDUALLY

	#600

	  reset = 0;
	
	
	  // store 5 -> reg 1

	  req3_cmd = 9;
	req3_d1 = 0;
	req3_d2 = 0;
	req3_data = 5 ;
	req3_r1 = 1;
	req3_tag = 0;

	#200

	req3_cmd = 0;
	req3_d1 = 0;
	req3_d2 = 0;
	req3_data = 0 ;
	req3_r1 = 0;
	req3_tag = 0;

	#200 

	  // store 10 -> reg 2
	  
	req3_cmd = 9;
	req3_d1 = 0;
	req3_d2 = 0;
	req3_data = 10 ;
	req3_r1 = 2;
	req3_tag = 1;
	
	#200

	req3_cmd = 0;
	req3_d1 = 0;
	req3_d2 = 0;
	req3_data = 0 ;
	req3_r1 = 0;
	req3_tag = 0;

	#200

	  // reg 1 (5) + reg 2 (10) -> reg 4
	  
	req3_cmd = 1;
	req3_d1 = 1;
	req3_d2 = 2;
	req3_data = 0 ;
	req3_r1 = 4;
	req3_tag = 2;
		
	#200

	req3_cmd = 0;
	req3_d1 = 0;
	req3_d2 = 0;
	req3_data = 0 ;
	req3_r1 = 0;
	req3_tag = 0;

	#200

	  // fetch reg 4 (15)
	  
	req3_cmd = 10;
	req3_d1 = 4;
	req3_d2 = 0;
	req3_data = 0 ;
	req3_r1 = 0;
	req3_tag = 3;
			
	#200

	  req3_cmd = 0;
	req3_d1 = 0;
	req3_d2 = 0;
	req3_data = 0 ;
	req3_r1 = 0;
	req3_tag = 0;

	#200
 
	  // reg 2 (10) - reg 1 (5) -> reg 5
	  
	  req3_cmd = 2;
	req3_d1 = 2;
	req3_d2 = 1;
	req3_data = 0 ;
	req3_r1 = 5;
	req3_tag = 0;

	#200

	req3_cmd = 0;
	req3_d1 = 0;
	req3_d2 = 0;
	req3_data = 0 ;
	req3_r1 = 0;
	req3_tag = 0;

	#200

	  // fetch reg 5
		  
	req3_cmd = 10;
	req3_d1 = 5;
	req3_d2 = 0;
	req3_data = 0 ;
	req3_r1 = 4;
	req3_tag = 1;

	#200

	req3_cmd = 0;
	req3_d1 = 0;
	req3_d2 = 0;
	req3_data = 0 ;
	req3_r1 = 0;
	req3_tag = 0;
	
	#200

	  // reg 2 ('b1010) << reg 1 (5) -> reg 6
		  
	  req3_cmd = 5;
	req3_d1 = 2;
	req3_d2 = 1;
	req3_data = 0 ;
	req3_r1 = 6;
	req3_tag = 2;

		#200

	req3_cmd = 0;
	req3_d1 = 0;
	req3_d2 = 0;
	req3_data = 0 ;
	req3_r1 = 0;
	req3_tag = 0;
	
	#200

	  // fetch reg 6 ('b101000000)
	  
	req3_cmd = 10;
	req3_d1 = 6;
	req3_d2 = 0;
	req3_data = 0 ;
	req3_r1 = 0;
	req3_tag = 3;

	#200

	req3_cmd = 0;
	req3_d1 = 0;
	req3_d2 = 0;
	req3_data = 0 ;
	req3_r1 = 0;
	req3_tag = 0;

	#200

	  // reg 2 (10) >> reg 1 (5) -> reg 7
	  
	  req3_cmd = 6;
	req3_d1 = 2;
	req3_d2 = 1;
	req3_data = 0 ;
	req3_r1 = 7;
	req3_tag = 0;

	#200

	req3_cmd = 0;
	req3_d1 = 0;
	req3_d2 = 0;
	req3_data = 0 ;
	req3_r1 = 0;
	req3_tag = 0;
		
	#200

	  // fetch reg 7 ( 0 )
	  
	req3_cmd = 10;
	req3_d1 = 7;
	req3_d2 = 0;
	req3_data = 0 ;
	req3_r1 = 0;
	req3_tag = 1;
		
	#200

	req3_cmd = 0;
	req3_d1 = 0;
	req3_d2 = 0;
	req3_data = 0 ;
	req3_r1 = 0;
	req3_tag = 0;

	#200
	  
	// TEST REQUESTOR 4 INDIVIDUALLY
	
	#1000 

	  reset = 1;

	#600 reset = 0;
		  
	  // store 5 -> reg 1

	  req4_cmd = 9;
	req4_d1 = 0;
	req4_d2 = 0;
	req4_data = 5 ;
	req4_r1 = 1;
	req4_tag = 0;

	#200

	req4_cmd = 0;
	req4_d1 = 0;
	req4_d2 = 0;
	req4_data = 0 ;
	req4_r1 = 0;
	req4_tag = 0;

	#200 

	  // store 10 -> reg 2
	  
	req4_cmd = 9;
	req4_d1 = 0;
	req4_d2 = 0;
	req4_data = 10 ;
	req4_r1 = 2;
	req4_tag = 1;
	
	#200

	req4_cmd = 0;
	req4_d1 = 0;
	req4_d2 = 0;
	req4_data = 0 ;
	req4_r1 = 0;
	req4_tag = 0;

	#200

	  // reg 1 (5) + reg 2 (10) -> reg 4
	  
	req4_cmd = 1;
	req4_d1 = 1;
	req4_d2 = 2;
	req4_data = 0 ;
	req4_r1 = 4;
	req4_tag = 2;
		
	#200

	req4_cmd = 0;
	req4_d1 = 0;
	req4_d2 = 0;
	req4_data = 0 ;
	req4_r1 = 0;
	req4_tag = 0;

	#200

	  // fetch reg 4 (15)
	  
	req4_cmd = 10;
	req4_d1 = 4;
	req4_d2 = 0;
	req4_data = 0 ;
	req4_r1 = 0;
	req4_tag = 3;
			
	#200

	  req4_cmd = 0;
	req4_d1 = 0;
	req4_d2 = 0;
	req4_data = 0 ;
	req4_r1 = 0;
	req4_tag = 0;

	#200

	  // reg 2 (10) - reg 1 (5) -> reg 5
	  
	  req4_cmd = 2;
	req4_d1 = 2;
	req4_d2 = 1;
	req4_data = 0 ;
	req4_r1 = 5;
	req4_tag = 0;

	#200

	req4_cmd = 0;
	req4_d1 = 0;
	req4_d2 = 0;
	req4_data = 0 ;
	req4_r1 = 0;
	req4_tag = 0;

	#200

	  // fetch reg 5
		  
	req4_cmd = 10;
	req4_d1 = 5;
	req4_d2 = 0;
	req4_data = 0 ;
	req4_r1 = 4;
	req4_tag = 1;

	#200

	req4_cmd = 0;
	req4_d1 = 0;
	req4_d2 = 0;
	req4_data = 0 ;
	req4_r1 = 0;
	req4_tag = 0;
	
	#200

	  // reg 2 ('b1010) << reg 1 (5) -> reg 6
		  
	  req4_cmd = 5;
	req4_d1 = 2;
	req4_d2 = 1;
	req4_data = 0 ;
	req4_r1 = 6;
	req4_tag = 2;

		#200

	req4_cmd = 0;
	req4_d1 = 0;
	req4_d2 = 0;
	req4_data = 0 ;
	req4_r1 = 0;
	req4_tag = 0;
	
	#200

	  // fetch reg 6 ('b101000000)
	  
	req4_cmd = 10;
	req4_d1 = 6;
	req4_d2 = 0;
	req4_data = 0 ;
	req4_r1 = 0;
	req4_tag = 3;

	#200

	req4_cmd = 0;
	req4_d1 = 0;
	req4_d2 = 0;
	req4_data = 0 ;
	req4_r1 = 0;
	req4_tag = 0;

	#200

	  // reg 2 (10) >> reg 1 (5) -> reg 7
	  
	  req4_cmd = 6;
	req4_d1 = 2;
	req4_d2 = 1;
	req4_data = 0 ;
	req4_r1 = 7;
	req4_tag = 0;

	#200

	req4_cmd = 0;
	req4_d1 = 0;
	req4_d2 = 0;
	req4_data = 0 ;
	req4_r1 = 0;
	req4_tag = 0;
		
	#200

	  // fetch reg 7 ( 0 )
	  
	req4_cmd = 10;
	req4_d1 = 7;
	req4_d2 = 0;
	req4_data = 0 ;
	req4_r1 = 0;
	req4_tag = 1;
		
	#200

	req4_cmd = 0;
	req4_d1 = 0;
	req4_d2 = 0;
	req4_data = 0 ;
	req4_r1 = 0;
	req4_tag = 0;

	#200



	#2000 $stop;

     end // initial begin

   always
     @ (reset or c_clk or req1_cmd or req1_data or req2_cmd or req2_data or req3_cmd or req3_data or req4_cmd or req4_data) begin
	
	$display ("%t: r:%b \n 
1c:%d,1d1:%d,1d2:%d,1d:%d,1r:%d,1t:%d
2c:%d,2d1:%d,2d2:%d,2d:%d,2r:%d,2t:%d
3c:%d,3d1:%d,3d2:%d,3d:%d,3r:%d,3t:%d
4c:%d,4d1:%d,4d2:%d,4d:%d,4r:%d,4t:%d  
1r:%d,1d:%d,1t:%d  
2r:%d,2d:%d,2t:%d  
3r:%d,3d:%d,3t:%d  
4r:%d,4d:%d,4t:%d \n\n", 

$time, reset, 
req1_cmd, req1_d1, req1_d2, req1_data, req1_r1, req1_tag,
req2_cmd, req2_d1, req2_d2, req2_data, req2_r1, req2_tag,
req3_cmd, req3_d1, req3_d2, req3_data, req3_r1, req3_tag,
req4_cmd, req4_d1, req4_d2, req4_data, req4_r1, req4_tag,
out1_resp, out1_data, out1_tag,
out2_resp, out2_data, out2_tag,
out3_resp, out3_data, out3_tag,
out4_resp, out4_data, out4_tag);
	
     end

endmodule // test

module atest1;

//=======================================================================
// This tests for the first bug.  Although the add which follows the branch
// results in a correct response of 11 (skipped), the resultant sum is still
// written to the register.  It should not write the sum to the register if it
// is truly skipped.

   wire [0:1] out1_resp, out1_tag, out2_resp, out2_tag, out3_resp, out3_tag, out4_resp, out4_tag;
   wire [0:31] out1_data, out2_data, out3_data, out4_data;
   wire 	 scan_out;
   
   reg [0:1] req1_tag, req2_tag, req3_tag, req4_tag;
   reg       a_clk, b_clk, c_clk, reset, scan_in;
   reg [0:3] req1_cmd, req1_d1, req1_d2, req1_r1, req2_cmd, req2_d1, req2_d2, req2_r1, req3_cmd, req3_d1, req3_d2, req3_r1, req4_cmd, req4_d1, req4_d2, req4_r1;
   reg [0:31] req1_data, req2_data, req3_data, req4_data;
      
   wire [0:15] add_shift_branch_data;
   wire [0:3]  adder_cmd, adder_out_cmd;
   wire [0:4]  adder_follow_branch;
   wire [0:31] adder_out_data1, adder_out_data2, adder_out_data3, adder_out_data4;
   wire [0:1]  adder_out_resp1, adder_out_resp2, adder_out_resp3, adder_out_resp4, adder_out_tag1, adder_out_tag2, adder_out_tag3, adder_out_tag4;
   wire        adder_overflow;
   wire [0:3]  adder_read_adr1, adder_read_adr2; 
   wire [0:63] adder_read_d1, adder_read_d2; 
   wire        adder_read_valid1, adder_read_valid2;
   wire [0:63] adder_result;
   wire [0:4]  adder_result_reg;
   wire [0:3]  adder_tag, adder_write_adr;
   wire [0:31] adder_write_data;
   wire        adder_write_valid;
   wire [0:3]  hold1_cmd, hold1_d1, hold1_d2;
   wire [0:31] hold1_data;
   wire [0:3]  hold1_r1;
   wire [0:1]  hold1_tag;
   wire [0:3]  hold2_cmd, hold2_d1, hold2_d2;
   wire [0:31] hold2_data;
   wire [0:3]  hold2_r1;
   wire [0:1]  hold2_tag;
   wire [0:3]  hold3_cmd, hold3_d1, hold3_d2;
   wire [0:31] hold3_data;
   wire [0:3]  hold3_r1;
   wire [0:1]  hold3_tag;
   wire [0:3]  hold4_cmd;
   wire [0:3]  hold4_d1,  hold4_d2;
   wire [0:31] hold4_data;
   wire [0:3]  hold4_r1;
   wire [0:1]  hold4_tag;
   wire port1_invalid_op;
   wire [0:1] port1_invalid_tag;
   wire       port2_invalid_op;
   wire [0:1] port2_invalid_tag;
   wire       port3_invalid_op;
   wire [0:1] port3_invalid_tag;
   wire       port4_invalid_op;
   wire [0:1] port4_invalid_tag;
   wire [0:3] prio_adder_cmd;
   wire [0:4] prio_adder_data1, prio_adder_data2;
   wire [0:4] prio_adder_follow_branch;
   wire       prio_adder_out_vld;
   wire [0:4] prio_adder_result;
   wire [0:3] prio_adder_tag,  prio_shift_cmd;
   wire [0:31] prio_shift_data;
   wire [0:4]  prio_shift_data1,  prio_shift_data2;
   wire [0:4]  prio_shift_follow_branch;
   wire        prio_shift_out_vld;
   wire [0:4]  prio_shift_result;
   wire [0:3]  prio_shift_tag;
   wire        scan_connect1, scan_connect10, scan_connect11, scan_connect12, scan_connect2, scan_connect3, scan_connect4, scan_connect5, scan_connect6, scan_connect7, scan_connect8, scan_connect9;
   wire [0:3]  shift_cmd;
   wire [0:4]  shift_follow_branch;
   wire [0:3]  shift_out_cmd;
   wire [0:31] shift_out_data1, shift_out_data2, shift_out_data3, shift_out_data4;
   wire [0:1]  shift_out_resp1;
   wire [0:1]  shift_out_resp2, shift_out_resp3, shift_out_resp4, shift_out_tag1, shift_out_tag2, shift_out_tag3, shift_out_tag4;
   wire [0:3]  shift_read_adr1,  shift_read_adr2;
   wire [0:63] shift_read_d1, shift_read_d2;
   wire        shift_read_valid1,  shift_read_valid2;
   wire [0:63] shift_result;
   wire [0:4]  shift_result_reg;
   wire [0:3]  shift_tag, shift_write_adr;
   wire [0:31] shift_write_data;
   wire        shift_write_valid, store_data_valid;
   wire [0:63] store_val;
   
   calc3_top ( out1_data, out1_resp, out1_tag, out2_data, out2_resp, out2_tag, out3_data, out3_resp, out3_tag, out4_data, out4_resp, out4_tag, scan_out, a_clk, b_clk, c_clk, req1_cmd, req1_d1, req1_d2, req1_data, req1_r1, req1_tag, req2_cmd, req2_d1, req2_d2, req2_data, req2_r1, req2_tag, req3_cmd, req3_d1, req3_d2, req3_data, req3_r1, req3_tag, req4_cmd, req4_d1, req4_d2, req4_data, req4_r1, req4_tag, reset, scan_in);
      
   initial 
     begin
	c_clk = 0;
	a_clk = 0;
	b_clk = 0;
	scan_in = 0;
	
     end
   
	
   always #100 c_clk = ~c_clk;
   
   initial
     begin

	reset = 1;

	
	
	#600 

	  reset = 0;
	req1_cmd = 0;
	req1_d1 = 0;
	req1_d2 = 0;
	req1_data = 0 ;
	req1_r1 = 0;
	req1_tag = 0;
	req2_cmd = 0;
	req2_d1 = 0;
	req2_d2 = 0;
	req2_data = 0 ;
	req2_r1 = 0;
	req2_tag = 0;
	req3_cmd = 0;
	req3_d1 = 0;
	req3_d2 = 0;
	req3_data = 0 ;
	req3_r1 = 0;
	req3_tag = 0;
	req4_cmd = 0;
	req4_d1 = 0;
	req4_d2 = 0;
	req4_data = 0 ;
	req4_r1 = 0;
	req4_tag = 0;
	
	#200

	  req1_cmd = 9;
	req1_d1 = 0;
	req1_d2 = 0;
	req1_data = 1 ;
	req1_r1 = 1;
	req1_tag = 0;
	
	#200 

	req1_cmd = 0;
	req1_d1 = 0;
	req1_d2 = 0;
	req1_data = 0 ;
	req1_r1 = 0;
	req1_tag = 0;

	#200

	req1_cmd = 12;
	req1_d1 = 0;
	req1_d2 = 0;
	req1_data = 0 ;
	req1_r1 = 0;
	req1_tag = 1;

	#200
	req1_cmd = 0;
	req1_d1 = 0;
	req1_d2 = 0;
	req1_data = 0 ;
	req1_r1 = 0;
	req1_tag = 0;

	#200
	  req1_cmd = 1;
	req1_d1 = 0;
	req1_d2 = 1;
	req1_data = 0 ;
	req1_r1 = 4;
	req1_tag = 2;

	#200

	  req1_cmd = 0;
	req1_d1 = 0;
	req1_d2 = 0;
	req1_data = 0 ;
	req1_r1 = 0;
	req1_tag = 0;
		
	#2000 $stop;

     end // initial begin

   always
     @ (reset or c_clk or req1_cmd or req1_data or req2_cmd or req2_data or req3_cmd or req3_data or req4_cmd or req4_data) begin
	
	$display ("%t: r:%b \n 
1c:%d,1d1:%d,1d2:%d,1d:%d,1r:%d,1t:%d
2c:%d,2d1:%d,2d2:%d,2d:%d,2r:%d,2t:%d
3c:%d,3d1:%d,3d2:%d,3d:%d,3r:%d,3t:%d
4c:%d,4d1:%d,4d2:%d,4d:%d,4r:%d,4t:%d  
1r:%d,1d:%d,1t:%d  
2r:%d,2d:%d,2t:%d  
3r:%d,3d:%d,3t:%d  
4r:%d,4d:%d,4t:%d \n\n", 

$time, reset, 
req1_cmd, req1_d1, req1_d2, req1_data, req1_r1, req1_tag,
req2_cmd, req2_d1, req2_d2, req2_data, req2_r1, req2_tag,
req3_cmd, req3_d1, req3_d2, req3_data, req3_r1, req3_tag,
req4_cmd, req4_d1, req4_d2, req4_data, req4_r1, req4_tag,
out1_resp, out1_data, out1_tag,
out2_resp, out2_data, out2_tag,
out3_resp, out3_data, out3_tag,
out4_resp, out4_data, out4_tag);
	
     end


endmodule // test1


module atest2;

   //==============================================================================
   // This tests bug 2, which fixes bit 13 of register 13 to 0 if register 13 is used as d2
   // in an add/subtract/branch command.  Tihs bug will cause incorrect results from these
   // commands if the stuck bit is supposed to be a 1 instead of a 0.
   
   wire [0:1] out1_resp, out1_tag, out2_resp, out2_tag, out3_resp, out3_tag, out4_resp, out4_tag;
   wire [0:31] out1_data, out2_data, out3_data, out4_data;
   wire 	 scan_out;
   
   reg [0:1] req1_tag, req2_tag, req3_tag, req4_tag;
   reg       a_clk, b_clk, c_clk, reset, scan_in;
   reg [0:3] req1_cmd, req1_d1, req1_d2, req1_r1, req2_cmd, req2_d1, req2_d2, req2_r1, req3_cmd, req3_d1, req3_d2, req3_r1, req4_cmd, req4_d1, req4_d2, req4_r1;
   reg [0:31] req1_data, req2_data, req3_data, req4_data;
      
   wire [0:15] add_shift_branch_data;
   wire [0:3]  adder_cmd, adder_out_cmd;
   wire [0:4]  adder_follow_branch;
   wire [0:31] adder_out_data1, adder_out_data2, adder_out_data3, adder_out_data4;
   wire [0:1]  adder_out_resp1, adder_out_resp2, adder_out_resp3, adder_out_resp4, adder_out_tag1, adder_out_tag2, adder_out_tag3, adder_out_tag4;
   wire        adder_overflow;
   wire [0:3]  adder_read_adr1, adder_read_adr2; 
   wire [0:63] adder_read_d1, adder_read_d2; 
   wire        adder_read_valid1, adder_read_valid2;
   wire [0:63] adder_result;
   wire [0:4]  adder_result_reg;
   wire [0:3]  adder_tag, adder_write_adr;
   wire [0:31] adder_write_data;
   wire        adder_write_valid;
   wire [0:3]  hold1_cmd, hold1_d1, hold1_d2;
   wire [0:31] hold1_data;
   wire [0:3]  hold1_r1;
   wire [0:1]  hold1_tag;
   wire [0:3]  hold2_cmd, hold2_d1, hold2_d2;
   wire [0:31] hold2_data;
   wire [0:3]  hold2_r1;
   wire [0:1]  hold2_tag;
   wire [0:3]  hold3_cmd, hold3_d1, hold3_d2;
   wire [0:31] hold3_data;
   wire [0:3]  hold3_r1;
   wire [0:1]  hold3_tag;
   wire [0:3]  hold4_cmd;
   wire [0:3]  hold4_d1,  hold4_d2;
   wire [0:31] hold4_data;
   wire [0:3]  hold4_r1;
   wire [0:1]  hold4_tag;
   wire port1_invalid_op;
   wire [0:1] port1_invalid_tag;
   wire       port2_invalid_op;
   wire [0:1] port2_invalid_tag;
   wire       port3_invalid_op;
   wire [0:1] port3_invalid_tag;
   wire       port4_invalid_op;
   wire [0:1] port4_invalid_tag;
   wire [0:3] prio_adder_cmd;
   wire [0:4] prio_adder_data1, prio_adder_data2;
   wire [0:4] prio_adder_follow_branch;
   wire       prio_adder_out_vld;
   wire [0:4] prio_adder_result;
   wire [0:3] prio_adder_tag,  prio_shift_cmd;
   wire [0:31] prio_shift_data;
   wire [0:4]  prio_shift_data1,  prio_shift_data2;
   wire [0:4]  prio_shift_follow_branch;
   wire        prio_shift_out_vld;
   wire [0:4]  prio_shift_result;
   wire [0:3]  prio_shift_tag;
   wire        scan_connect1, scan_connect10, scan_connect11, scan_connect12, scan_connect2, scan_connect3, scan_connect4, scan_connect5, scan_connect6, scan_connect7, scan_connect8, scan_connect9;
   wire [0:3]  shift_cmd;
   wire [0:4]  shift_follow_branch;
   wire [0:3]  shift_out_cmd;
   wire [0:31] shift_out_data1, shift_out_data2, shift_out_data3, shift_out_data4;
   wire [0:1]  shift_out_resp1;
   wire [0:1]  shift_out_resp2, shift_out_resp3, shift_out_resp4, shift_out_tag1, shift_out_tag2, shift_out_tag3, shift_out_tag4;
   wire [0:3]  shift_read_adr1,  shift_read_adr2;
   wire [0:63] shift_read_d1, shift_read_d2;
   wire        shift_read_valid1,  shift_read_valid2;
   wire [0:63] shift_result;
   wire [0:4]  shift_result_reg;
   wire [0:3]  shift_tag, shift_write_adr;
   wire [0:31] shift_write_data;
   wire        shift_write_valid, store_data_valid;
   wire [0:63] store_val;
   
   calc3_top ( out1_data, out1_resp, out1_tag, out2_data, out2_resp, out2_tag, out3_data, out3_resp, out3_tag, out4_data, out4_resp, out4_tag, scan_out, a_clk, b_clk, c_clk, req1_cmd, req1_d1, req1_d2, req1_data, req1_r1, req1_tag, req2_cmd, req2_d1, req2_d2, req2_data, req2_r1, req2_tag, req3_cmd, req3_d1, req3_d2, req3_data, req3_r1, req3_tag, req4_cmd, req4_d1, req4_d2, req4_data, req4_r1, req4_tag, reset, scan_in);
      
   initial 
     begin
	c_clk = 0;
	a_clk = 0;
	b_clk = 0;
	scan_in = 0;
	
     end
   
	
   always #100 c_clk = ~c_clk;
   
   initial
     begin

	reset = 1;

	
	
	#600 

	  reset = 0;
	req1_cmd = 0;
	req1_d1 = 0;
	req1_d2 = 0;
	req1_data = 0 ;
	req1_r1 = 0;
	req1_tag = 0;
	req2_cmd = 0;
	req2_d1 = 0;
	req2_d2 = 0;
	req2_data = 0 ;
	req2_r1 = 0;
	req2_tag = 0;
	req3_cmd = 0;
	req3_d1 = 0;
	req3_d2 = 0;
	req3_data = 0 ;
	req3_r1 = 0;
	req3_tag = 0;
	req4_cmd = 0;
	req4_d1 = 0;
	req4_d2 = 0;
	req4_data = 0 ;
	req4_r1 = 0;
	req4_tag = 0;
	
	#200

	req1_cmd = 9;
	req1_d1 = 0;
	req1_d2 = 0;
	req1_data = {32{1'b1}} ;
	req1_r1 = 13;
	req1_tag = 0;
	
	#200 

	req1_cmd = 0;
	req1_d1 = 0;
	req1_d2 = 0;
	req1_data = 0 ;
	req1_r1 = 0;
	req1_tag = 0;

	#200

	req1_cmd = 1;
	req1_d1 = 0;
	req1_d2 = 13;
	req1_data = 0 ;
	req1_r1 = 4;
	req1_tag = 1;

	#200
	req1_cmd = 0;
	req1_d1 = 0;
	req1_d2 = 0;
	req1_data = 0 ;
	req1_r1 = 0;
	req1_tag = 0;

	#200
	  req1_cmd = 10;
	req1_d1 = 4;
	req1_d2 = 0;
	req1_data = 0 ;
	req1_r1 = 0;
	req1_tag = 2;

	#200

	  req1_cmd = 0;
	req1_d1 = 0;
	req1_d2 = 0;
	req1_data = 0 ;
	req1_r1 = 0;
	req1_tag = 0;
		
	#2000 $stop;

     end // initial begin

   always
     @ (reset or c_clk or req1_cmd or req1_data or req2_cmd or req2_data or req3_cmd or req3_data or req4_cmd or req4_data) begin
	
	$display ("%t: r:%b \n 
1c:%d,1d1:%d,1d2:%d,1d:%d,1r:%d,1t:%d
2c:%d,2d1:%d,2d2:%d,2d:%d,2r:%d,2t:%d
3c:%d,3d1:%d,3d2:%d,3d:%d,3r:%d,3t:%d
4c:%d,4d1:%d,4d2:%d,4d:%d,4r:%d,4t:%d  
1r:%d,1d:%d,1t:%d  
2r:%d,2d:%d,2t:%d  
3r:%d,3d:%d,3t:%d  
4r:%d,4d:%d,4t:%d \n\n", 

$time, reset, 
req1_cmd, req1_d1, req1_d2, req1_data, req1_r1, req1_tag,
req2_cmd, req2_d1, req2_d2, req2_data, req2_r1, req2_tag,
req3_cmd, req3_d1, req3_d2, req3_data, req3_r1, req3_tag,
req4_cmd, req4_d1, req4_d2, req4_data, req4_r1, req4_tag,
out1_resp, out1_data, out1_tag,
out2_resp, out2_data, out2_tag,
out3_resp, out3_data, out3_tag,
out4_resp, out4_data, out4_tag);
	
     end


endmodule // test2

module atest3;

   //=====================================================================
   // This test shows that an overflowed value will still write to a result register
   // even though it is not supposed to

   wire [0:1] out1_resp, out1_tag, out2_resp, out2_tag, out3_resp, out3_tag, out4_resp, out4_tag;
   wire [0:31] out1_data, out2_data, out3_data, out4_data;
   wire 	 scan_out;
   
   reg [0:1] req1_tag, req2_tag, req3_tag, req4_tag;
   reg       a_clk, b_clk, c_clk, reset, scan_in;
   reg [0:3] req1_cmd, req1_d1, req1_d2, req1_r1, req2_cmd, req2_d1, req2_d2, req2_r1, req3_cmd, req3_d1, req3_d2, req3_r1, req4_cmd, req4_d1, req4_d2, req4_r1;
   reg [0:31] req1_data, req2_data, req3_data, req4_data;
      
   wire [0:15] add_shift_branch_data;
   wire [0:3]  adder_cmd, adder_out_cmd;
   wire [0:4]  adder_follow_branch;
   wire [0:31] adder_out_data1, adder_out_data2, adder_out_data3, adder_out_data4;
   wire [0:1]  adder_out_resp1, adder_out_resp2, adder_out_resp3, adder_out_resp4, adder_out_tag1, adder_out_tag2, adder_out_tag3, adder_out_tag4;
   wire        adder_overflow;
   wire [0:3]  adder_read_adr1, adder_read_adr2; 
   wire [0:63] adder_read_d1, adder_read_d2; 
   wire        adder_read_valid1, adder_read_valid2;
   wire [0:63] adder_result;
   wire [0:4]  adder_result_reg;
   wire [0:3]  adder_tag, adder_write_adr;
   wire [0:31] adder_write_data;
   wire        adder_write_valid;
   wire [0:3]  hold1_cmd, hold1_d1, hold1_d2;
   wire [0:31] hold1_data;
   wire [0:3]  hold1_r1;
   wire [0:1]  hold1_tag;
   wire [0:3]  hold2_cmd, hold2_d1, hold2_d2;
   wire [0:31] hold2_data;
   wire [0:3]  hold2_r1;
   wire [0:1]  hold2_tag;
   wire [0:3]  hold3_cmd, hold3_d1, hold3_d2;
   wire [0:31] hold3_data;
   wire [0:3]  hold3_r1;
   wire [0:1]  hold3_tag;
   wire [0:3]  hold4_cmd;
   wire [0:3]  hold4_d1,  hold4_d2;
   wire [0:31] hold4_data;
   wire [0:3]  hold4_r1;
   wire [0:1]  hold4_tag;
   wire port1_invalid_op;
   wire [0:1] port1_invalid_tag;
   wire       port2_invalid_op;
   wire [0:1] port2_invalid_tag;
   wire       port3_invalid_op;
   wire [0:1] port3_invalid_tag;
   wire       port4_invalid_op;
   wire [0:1] port4_invalid_tag;
   wire [0:3] prio_adder_cmd;
   wire [0:4] prio_adder_data1, prio_adder_data2;
   wire [0:4] prio_adder_follow_branch;
   wire       prio_adder_out_vld;
   wire [0:4] prio_adder_result;
   wire [0:3] prio_adder_tag,  prio_shift_cmd;
   wire [0:31] prio_shift_data;
   wire [0:4]  prio_shift_data1,  prio_shift_data2;
   wire [0:4]  prio_shift_follow_branch;
   wire        prio_shift_out_vld;
   wire [0:4]  prio_shift_result;
   wire [0:3]  prio_shift_tag;
   wire        scan_connect1, scan_connect10, scan_connect11, scan_connect12, scan_connect2, scan_connect3, scan_connect4, scan_connect5, scan_connect6, scan_connect7, scan_connect8, scan_connect9;
   wire [0:3]  shift_cmd;
   wire [0:4]  shift_follow_branch;
   wire [0:3]  shift_out_cmd;
   wire [0:31] shift_out_data1, shift_out_data2, shift_out_data3, shift_out_data4;
   wire [0:1]  shift_out_resp1;
   wire [0:1]  shift_out_resp2, shift_out_resp3, shift_out_resp4, shift_out_tag1, shift_out_tag2, shift_out_tag3, shift_out_tag4;
   wire [0:3]  shift_read_adr1,  shift_read_adr2;
   wire [0:63] shift_read_d1, shift_read_d2;
   wire        shift_read_valid1,  shift_read_valid2;
   wire [0:63] shift_result;
   wire [0:4]  shift_result_reg;
   wire [0:3]  shift_tag, shift_write_adr;
   wire [0:31] shift_write_data;
   wire        shift_write_valid, store_data_valid;
   wire [0:63] store_val;
   
   calc3_top ( out1_data, out1_resp, out1_tag, out2_data, out2_resp, out2_tag, out3_data, out3_resp, out3_tag, out4_data, out4_resp, out4_tag, scan_out, a_clk, b_clk, c_clk, req1_cmd, req1_d1, req1_d2, req1_data, req1_r1, req1_tag, req2_cmd, req2_d1, req2_d2, req2_data, req2_r1, req2_tag, req3_cmd, req3_d1, req3_d2, req3_data, req3_r1, req3_tag, req4_cmd, req4_d1, req4_d2, req4_data, req4_r1, req4_tag, reset, scan_in);
      
   initial 
     begin
	c_clk = 0;
	a_clk = 0;
	b_clk = 0;
	scan_in = 0;
	
     end
   
	
   always #100 c_clk = ~c_clk;
   
   initial
     begin

	reset = 1;

	
	
	#600 

	  reset = 0;
	req1_cmd = 0;
	req1_d1 = 0;
	req1_d2 = 0;
	req1_data = 0 ;
	req1_r1 = 0;
	req1_tag = 0;
	req2_cmd = 0;
	req2_d1 = 0;
	req2_d2 = 0;
	req2_data = 0 ;
	req2_r1 = 0;
	req2_tag = 0;
	req3_cmd = 0;
	req3_d1 = 0;
	req3_d2 = 0;
	req3_data = 0 ;
	req3_r1 = 0;
	req3_tag = 0;
	req4_cmd = 0;
	req4_d1 = 0;
	req4_d2 = 0;
	req4_data = 0 ;
	req4_r1 = 0;
	req4_tag = 0;
	
	#200

	req1_cmd = 9;
	req1_d1 = 0;
	req1_d2 = 0;
	req1_data = {32{1'b1}} ;
	req1_r1 = 0;
	req1_tag = 0;
	
	#200 

	req1_cmd = 0;
	req1_d1 = 0;
	req1_d2 = 0;
	req1_data = 0 ;
	req1_r1 = 0;
	req1_tag = 0;

	#200

	req1_cmd = 9;
	req1_d1 = 0;
	req1_d2 = 0;
	req1_data = {16{1'b1}} ;
	req1_r1 = 1;
	req1_tag = 1;

	#200
	req1_cmd = 0;
	req1_d1 = 0;
	req1_d2 = 0;
	req1_data = 0 ;
	req1_r1 = 0;
	req1_tag = 0;

	#200
	  req1_cmd = 1;
	req1_d1 = 0;
	req1_d2 = 1;
	req1_data = 0 ;
	req1_r1 = 4;
	req1_tag = 2;

	#200

	  req1_cmd = 0;
	req1_d1 = 0;
	req1_d2 = 0;
	req1_data = 0 ;
	req1_r1 = 0;
	req1_tag = 0;
		
	#2000 $stop;

     end // initial begin

   always
     @ (reset or c_clk or req1_cmd or req1_data or req2_cmd or req2_data or req3_cmd or req3_data or req4_cmd or req4_data) begin
	
	$display ("%t: r:%b \n 
1c:%d,1d1:%d,1d2:%d,1d:%d,1r:%d,1t:%d
2c:%d,2d1:%d,2d2:%d,2d:%d,2r:%d,2t:%d
3c:%d,3d1:%d,3d2:%d,3d:%d,3r:%d,3t:%d
4c:%d,4d1:%d,4d2:%d,4d:%d,4r:%d,4t:%d  
1r:%d,1d:%d,1t:%d  
2r:%d,2d:%d,2t:%d  
3r:%d,3d:%d,3t:%d  
4r:%d,4d:%d,4t:%d \n\n", 

		  $time, reset, 
		  req1_cmd, req1_d1, req1_d2, req1_data, req1_r1, req1_tag,
		  req2_cmd, req2_d1, req2_d2, req2_data, req2_r1, req2_tag,
		  req3_cmd, req3_d1, req3_d2, req3_data, req3_r1, req3_tag,
		  req4_cmd, req4_d1, req4_d2, req4_data, req4_r1, req4_tag,
		  out1_resp, out1_data, out1_tag,
		  out2_resp, out2_data, out2_tag,
		  out3_resp, out3_data, out3_tag,
		  out4_resp, out4_data, out4_tag);
	
     end
   
   
endmodule // test3


module atest4;

   wire [0:1] out1_resp, out1_tag, out2_resp, out2_tag, out3_resp, out3_tag, out4_resp, out4_tag;
   wire [0:31] out1_data, out2_data, out3_data, out4_data;
   wire 	 scan_out;
   
   reg [0:1] req1_tag, req2_tag, req3_tag, req4_tag;
   reg       a_clk, b_clk, c_clk, reset, scan_in;
   reg [0:3] req1_cmd, req1_d1, req1_d2, req1_r1, req2_cmd, req2_d1, req2_d2, req2_r1, req3_cmd, req3_d1, req3_d2, req3_r1, req4_cmd, req4_d1, req4_d2, req4_r1;
   reg [0:31] req1_data, req2_data, req3_data, req4_data;
      
   wire [0:15] add_shift_branch_data;
   wire [0:3]  adder_cmd, adder_out_cmd;
   wire [0:4]  adder_follow_branch;
   wire [0:31] adder_out_data1, adder_out_data2, adder_out_data3, adder_out_data4;
   wire [0:1]  adder_out_resp1, adder_out_resp2, adder_out_resp3, adder_out_resp4, adder_out_tag1, adder_out_tag2, adder_out_tag3, adder_out_tag4;
   wire        adder_overflow;
   wire [0:3]  adder_read_adr1, adder_read_adr2; 
   wire [0:63] adder_read_d1, adder_read_d2; 
   wire        adder_read_valid1, adder_read_valid2;
   wire [0:63] adder_result;
   wire [0:4]  adder_result_reg;
   wire [0:3]  adder_tag, adder_write_adr;
   wire [0:31] adder_write_data;
   wire        adder_write_valid;
   wire [0:3]  hold1_cmd, hold1_d1, hold1_d2;
   wire [0:31] hold1_data;
   wire [0:3]  hold1_r1;
   wire [0:1]  hold1_tag;
   wire [0:3]  hold2_cmd, hold2_d1, hold2_d2;
   wire [0:31] hold2_data;
   wire [0:3]  hold2_r1;
   wire [0:1]  hold2_tag;
   wire [0:3]  hold3_cmd, hold3_d1, hold3_d2;
   wire [0:31] hold3_data;
   wire [0:3]  hold3_r1;
   wire [0:1]  hold3_tag;
   wire [0:3]  hold4_cmd;
   wire [0:3]  hold4_d1,  hold4_d2;
   wire [0:31] hold4_data;
   wire [0:3]  hold4_r1;
   wire [0:1]  hold4_tag;
   wire port1_invalid_op;
   wire [0:1] port1_invalid_tag;
   wire       port2_invalid_op;
   wire [0:1] port2_invalid_tag;
   wire       port3_invalid_op;
   wire [0:1] port3_invalid_tag;
   wire       port4_invalid_op;
   wire [0:1] port4_invalid_tag;
   wire [0:3] prio_adder_cmd;
   wire [0:4] prio_adder_data1, prio_adder_data2;
   wire [0:4] prio_adder_follow_branch;
   wire       prio_adder_out_vld;
   wire [0:4] prio_adder_result;
   wire [0:3] prio_adder_tag,  prio_shift_cmd;
   wire [0:31] prio_shift_data;
   wire [0:4]  prio_shift_data1,  prio_shift_data2;
   wire [0:4]  prio_shift_follow_branch;
   wire        prio_shift_out_vld;
   wire [0:4]  prio_shift_result;
   wire [0:3]  prio_shift_tag;
   wire        scan_connect1, scan_connect10, scan_connect11, scan_connect12, scan_connect2, scan_connect3, scan_connect4, scan_connect5, scan_connect6, scan_connect7, scan_connect8, scan_connect9;
   wire [0:3]  shift_cmd;
   wire [0:4]  shift_follow_branch;
   wire [0:3]  shift_out_cmd;
   wire [0:31] shift_out_data1, shift_out_data2, shift_out_data3, shift_out_data4;
   wire [0:1]  shift_out_resp1;
   wire [0:1]  shift_out_resp2, shift_out_resp3, shift_out_resp4, shift_out_tag1, shift_out_tag2, shift_out_tag3, shift_out_tag4;
   wire [0:3]  shift_read_adr1,  shift_read_adr2;
   wire [0:63] shift_read_d1, shift_read_d2;
   wire        shift_read_valid1,  shift_read_valid2;
   wire [0:63] shift_result;
   wire [0:4]  shift_result_reg;
   wire [0:3]  shift_tag, shift_write_adr;
   wire [0:31] shift_write_data;
   wire        shift_write_valid, store_data_valid;
   wire [0:63] store_val;
   
   calc3_top ( out1_data, out1_resp, out1_tag, out2_data, out2_resp, out2_tag, out3_data, out3_resp, out3_tag, out4_data, out4_resp, out4_tag, scan_out, a_clk, b_clk, c_clk, req1_cmd, req1_d1, req1_d2, req1_data, req1_r1, req1_tag, req2_cmd, req2_d1, req2_d2, req2_data, req2_r1, req2_tag, req3_cmd, req3_d1, req3_d2, req3_data, req3_r1, req3_tag, req4_cmd, req4_d1, req4_d2, req4_data, req4_r1, req4_tag, reset, scan_in);
      
   initial 
     begin
	c_clk = 0;
	a_clk = 0;
	b_clk = 0;
	scan_in = 0;
	
     end
   
	
   always #100 c_clk = ~c_clk;
   
   initial
     begin

	reset = 1;

	
	
	#600 

	  reset = 0;
	req1_cmd = 0;
	req1_d1 = 0;
	req1_d2 = 0;
	req1_data = 0 ;
	req1_r1 = 0;
	req1_tag = 0;
	req2_cmd = 0;
	req2_d1 = 0;
	req2_d2 = 0;
	req2_data = 0 ;
	req2_r1 = 0;
	req2_tag = 0;
	req3_cmd = 0;
	req3_d1 = 0;
	req3_d2 = 0;
	req3_data = 0 ;
	req3_r1 = 0;
	req3_tag = 0;
	req4_cmd = 0;
	req4_d1 = 0;
	req4_d2 = 0;
	req4_data = 0 ;
	req4_r1 = 0;
	req4_tag = 0;
	
	#200

	req1_cmd = 9;
	req1_d1 = 0;
	req1_d2 = 0;
	req1_data = 1 ;
	req1_r1 = 0;
	req1_tag = 0;

	req2_cmd = 9;
	req2_d1 = 0;
	req2_d2 = 0;
	req2_data = 2 ;
	req2_r1 = 1;
	req2_tag = 0;

	
	#200 

	req1_cmd = 0;
	req1_d1 = 0;
	req1_d2 = 0;
	req1_data = 0 ;
	req1_r1 = 0;
	req1_tag = 0;

	#600

	req1_cmd = 1;
	req1_d1 = 0;
	req1_d2 = 1;
	req1_data = 0 ;
	req1_r1 = 8;
	req1_tag = 1;

	req2_cmd = 5;
	req2_d1 = 0;
	req2_d2 = 1;
	req2_data = 0 ;
	req2_r1 = 8;
	req2_tag = 1;

	#200
	  req1_cmd = 0;
	req1_d1 = 0;
	req1_d2 = 0;
	req1_data = 0 ;
	req1_r1 = 0;
	req1_tag = 0;

	  req1_cmd = 0;
	req1_d1 = 0;
	req1_d2 = 0;
	req1_data = 0 ;
	req1_r1 = 0;
	req1_tag = 0;
		
	#2000 $stop;

     end // initial begin

   always
     @ (reset or c_clk or req1_cmd or req1_data or req2_cmd or req2_data or req3_cmd or req3_data or req4_cmd or req4_data) begin
	
	$display ("%t: r:%b \n 
1c:%d,1d1:%d,1d2:%d,1d:%d,1r:%d,1t:%d
2c:%d,2d1:%d,2d2:%d,2d:%d,2r:%d,2t:%d
3c:%d,3d1:%d,3d2:%d,3d:%d,3r:%d,3t:%d
4c:%d,4d1:%d,4d2:%d,4d:%d,4r:%d,4t:%d  
1r:%d,1d:%d,1t:%d  
2r:%d,2d:%d,2t:%d  
3r:%d,3d:%d,3t:%d  
4r:%d,4d:%d,4t:%d \n\n", 

		  $time, reset, 
		  req1_cmd, req1_d1, req1_d2, req1_data, req1_r1, req1_tag,
		  req2_cmd, req2_d1, req2_d2, req2_data, req2_r1, req2_tag,
		  req3_cmd, req3_d1, req3_d2, req3_data, req3_r1, req3_tag,
		  req4_cmd, req4_d1, req4_d2, req4_data, req4_r1, req4_tag,
		  out1_resp, out1_data, out1_tag,
		  out2_resp, out2_data, out2_tag,
		  out3_resp, out3_data, out3_tag,
		  out4_resp, out4_data, out4_tag);
	
     end
   
   
endmodule // test4


module atest5;

   wire [0:1] out1_resp, out1_tag, out2_resp, out2_tag, out3_resp, out3_tag, out4_resp, out4_tag;
   wire [0:31] out1_data, out2_data, out3_data, out4_data;
   wire 	 scan_out;
   
   reg [0:1] req1_tag, req2_tag, req3_tag, req4_tag;
   reg       a_clk, b_clk, c_clk, reset, scan_in;
   reg [0:3] req1_cmd, req1_d1, req1_d2, req1_r1, req2_cmd, req2_d1, req2_d2, req2_r1, req3_cmd, req3_d1, req3_d2, req3_r1, req4_cmd, req4_d1, req4_d2, req4_r1;
   reg [0:31] req1_data, req2_data, req3_data, req4_data;
      
   wire [0:15] add_shift_branch_data;
   wire [0:3]  adder_cmd, adder_out_cmd;
   wire [0:4]  adder_follow_branch;
   wire [0:31] adder_out_data1, adder_out_data2, adder_out_data3, adder_out_data4;
   wire [0:1]  adder_out_resp1, adder_out_resp2, adder_out_resp3, adder_out_resp4, adder_out_tag1, adder_out_tag2, adder_out_tag3, adder_out_tag4;
   wire        adder_overflow;
   wire [0:3]  adder_read_adr1, adder_read_adr2; 
   wire [0:63] adder_read_d1, adder_read_d2; 
   wire        adder_read_valid1, adder_read_valid2;
   wire [0:63] adder_result;
   wire [0:4]  adder_result_reg;
   wire [0:3]  adder_tag, adder_write_adr;
   wire [0:31] adder_write_data;
   wire        adder_write_valid;
   wire [0:3]  hold1_cmd, hold1_d1, hold1_d2;
   wire [0:31] hold1_data;
   wire [0:3]  hold1_r1;
   wire [0:1]  hold1_tag;
   wire [0:3]  hold2_cmd, hold2_d1, hold2_d2;
   wire [0:31] hold2_data;
   wire [0:3]  hold2_r1;
   wire [0:1]  hold2_tag;
   wire [0:3]  hold3_cmd, hold3_d1, hold3_d2;
   wire [0:31] hold3_data;
   wire [0:3]  hold3_r1;
   wire [0:1]  hold3_tag;
   wire [0:3]  hold4_cmd;
   wire [0:3]  hold4_d1,  hold4_d2;
   wire [0:31] hold4_data;
   wire [0:3]  hold4_r1;
   wire [0:1]  hold4_tag;
   wire port1_invalid_op;
   wire [0:1] port1_invalid_tag;
   wire       port2_invalid_op;
   wire [0:1] port2_invalid_tag;
   wire       port3_invalid_op;
   wire [0:1] port3_invalid_tag;
   wire       port4_invalid_op;
   wire [0:1] port4_invalid_tag;
   wire [0:3] prio_adder_cmd;
   wire [0:4] prio_adder_data1, prio_adder_data2;
   wire [0:4] prio_adder_follow_branch;
   wire       prio_adder_out_vld;
   wire [0:4] prio_adder_result;
   wire [0:3] prio_adder_tag,  prio_shift_cmd;
   wire [0:31] prio_shift_data;
   wire [0:4]  prio_shift_data1,  prio_shift_data2;
   wire [0:4]  prio_shift_follow_branch;
   wire        prio_shift_out_vld;
   wire [0:4]  prio_shift_result;
   wire [0:3]  prio_shift_tag;
   wire        scan_connect1, scan_connect10, scan_connect11, scan_connect12, scan_connect2, scan_connect3, scan_connect4, scan_connect5, scan_connect6, scan_connect7, scan_connect8, scan_connect9;
   wire [0:3]  shift_cmd;
   wire [0:4]  shift_follow_branch;
   wire [0:3]  shift_out_cmd;
   wire [0:31] shift_out_data1, shift_out_data2, shift_out_data3, shift_out_data4;
   wire [0:1]  shift_out_resp1;
   wire [0:1]  shift_out_resp2, shift_out_resp3, shift_out_resp4, shift_out_tag1, shift_out_tag2, shift_out_tag3, shift_out_tag4;
   wire [0:3]  shift_read_adr1,  shift_read_adr2;
   wire [0:63] shift_read_d1, shift_read_d2;
   wire        shift_read_valid1,  shift_read_valid2;
   wire [0:63] shift_result;
   wire [0:4]  shift_result_reg;
   wire [0:3]  shift_tag, shift_write_adr;
   wire [0:31] shift_write_data;
   wire        shift_write_valid, store_data_valid;
   wire [0:63] store_val;
   
   calc3_top ( out1_data, out1_resp, out1_tag, out2_data, out2_resp, out2_tag, out3_data, out3_resp, out3_tag, out4_data, out4_resp, out4_tag, scan_out, a_clk, b_clk, c_clk, req1_cmd, req1_d1, req1_d2, req1_data, req1_r1, req1_tag, req2_cmd, req2_d1, req2_d2, req2_data, req2_r1, req2_tag, req3_cmd, req3_d1, req3_d2, req3_data, req3_r1, req3_tag, req4_cmd, req4_d1, req4_d2, req4_data, req4_r1, req4_tag, reset, scan_in);
      
   initial 
     begin
	c_clk = 0;
	a_clk = 0;
	b_clk = 0;
	scan_in = 0;
	
     end
   
	
   always #100 c_clk = ~c_clk;
   
   initial
     begin

	reset = 1;

	
	
	#600 

	  reset = 0;
	req1_cmd = 0;
	req1_d1 = 0;
	req1_d2 = 0;
	req1_data = 0 ;
	req1_r1 = 0;
	req1_tag = 0;
	req2_cmd = 0;
	req2_d1 = 0;
	req2_d2 = 0;
	req2_data = 0 ;
	req2_r1 = 0;
	req2_tag = 0;
	req3_cmd = 0;
	req3_d1 = 0;
	req3_d2 = 0;
	req3_data = 0 ;
	req3_r1 = 0;
	req3_tag = 0;
	req4_cmd = 0;
	req4_d1 = 0;
	req4_d2 = 0;
	req4_data = 0 ;
	req4_r1 = 0;
	req4_tag = 0;
	
	#200 

	  req1_cmd = 9;
	req1_d1 = 0;
	req1_d2 = 0;
	req1_data = 0 ;
	req1_r1 = 0;
	req1_tag = 0;
	  
	req2_cmd = 9;
	req2_d1 = 0;
	req2_d2 = 0;
	req2_data = 1 ;
	req2_r1 = 1;
	req2_tag = 0;

	req3_cmd = 9;
	req3_d1 = 0;
	req3_d2 = 0;
	req3_data = 2 ;
	req3_r1 = 2;
	req3_tag = 0;

	req4_cmd = 9;
	req4_d1 = 0;
	req4_d2 = 0;
	req4_data = 3 ;
	req4_r1 = 3;
	req4_tag = 0;
	
	#200 

	  req1_cmd = 0;
	req1_d1 = 0;
	req1_d2 = 0;
	req1_data = 0 ;
	req1_r1 = 0;
	req1_tag = 0;
	req2_cmd = 0;
	req2_d1 = 0;
	req2_d2 = 0;
	req2_data = 0 ;
	req2_r1 = 0;
	req2_tag = 0;
	req3_cmd = 0;
	req3_d1 = 0;
	req3_d2 = 0;
	req3_data = 0 ;
	req3_r1 = 0;
	req3_tag = 0;
	req4_cmd = 0;
	req4_d1 = 0;
	req4_d2 = 0;
	req4_data = 0 ;
	req4_r1 = 0;
	req4_tag = 0;
		
	#200

	req1_cmd = 1;
	req1_d1 = 8;
	req1_d2 = 1;
	req1_data = 0 ;
	req1_r1 = 8;
	req1_tag = 0;

	req2_cmd = 1;
	req2_d1 = 8;
	req2_d2 = 1;
	req2_data = 0 ;
	req2_r1 = 8;
	req2_tag = 0;

	req3_cmd = 1;
	req3_d1 = 8;
	req3_d2 = 1;
	req3_data = 0 ;
	req3_r1 = 8;
	req3_tag = 0;

	req4_cmd = 1;
	req4_d1 = 8;
	req4_d2 = 1;
	req4_data = 0 ;
	req4_r1 = 8;
	req4_tag = 0;
		
	#200

	req1_cmd = 0;
	req1_d1 = 0;
	req1_d2 = 0;
	req1_data = 0 ;
	req1_r1 = 0;
	req1_tag = 0;
	req2_cmd = 0;
	req2_d1 = 0;
	req2_d2 = 0;
	req2_data = 0 ;
	req2_r1 = 0;
	req2_tag = 0;
	req3_cmd = 0;
	req3_d1 = 0;
	req3_d2 = 0;
	req3_data = 0 ;
	req3_r1 = 0;
	req3_tag = 0;
	req4_cmd = 0;
	req4_d1 = 0;
	req4_d2 = 0;
	req4_data = 0 ;
	req4_r1 = 0;
	req4_tag = 0;

	#200

	  req1_cmd = 1;
	req1_d1 = 8;
	req1_d2 = 1;
	req1_data = 0 ;
	req1_r1 = 8;
	req1_tag = 1;
	
	req2_cmd = 1;
	req2_d1 = 8;
	req2_d2 = 1;
	req2_data = 0 ;
	req2_r1 = 8;
	req2_tag = 1;
	
	req3_cmd = 1;
	req3_d1 = 8;
	req3_d2 = 1;
	req3_data = 0 ;
	req3_r1 = 8;
	req3_tag = 1;
	
	req4_cmd = 1;
	req4_d1 = 8;
	req4_d2 = 1;
	req4_data = 0 ;
	req4_r1 = 8;
	req4_tag = 1;
		
	#200

	  req1_cmd = 0;
	req1_d1 = 0;
	req1_d2 = 0;
	req1_data = 0 ;
	req1_r1 = 0;
	req1_tag = 0;
	req2_cmd = 0;
	req2_d1 = 0;
	req2_d2 = 0;
	req2_data = 0 ;
	req2_r1 = 0;
	req2_tag = 0;
	req3_cmd = 0;
	req3_d1 = 0;
	req3_d2 = 0;
	req3_data = 0 ;
	req3_r1 = 0;
	req3_tag = 0;
	req4_cmd = 0;
	req4_d1 = 0;
	req4_d2 = 0;
	req4_data = 0 ;
	req4_r1 = 0;
	req4_tag = 0; 

	#200

	  req1_cmd = 1;
	req1_d1 = 8;
	req1_d2 = 1;
	req1_data = 0 ;
	req1_r1 = 8;
	req1_tag = 2;
	req2_cmd = 1;
	req2_d1 = 8;
	req2_d2 = 1;
	req2_data = 0 ;
	req2_r1 = 8;
	req2_tag = 2;
	req3_cmd = 1;
	req3_d1 = 8;
	req3_d2 = 1;
	req3_data = 0 ;
	req3_r1 = 8;
	req3_tag = 2;
	req4_cmd = 1;
	req4_d1 = 8;
	req4_d2 = 1;
	req4_data = 0 ;
	req4_r1 = 8;
	req4_tag = 2;

	#200

	  req1_cmd = 0;
	req1_d1 = 0;
	req1_d2 = 0;
	req1_data = 0 ;
	req1_r1 = 0;
	req1_tag = 0;
	req2_cmd = 0;
	req2_d1 = 0;
	req2_d2 = 0;
	req2_data = 0 ;
	req2_r1 = 0;
	req2_tag = 0;
	req3_cmd = 0;
	req3_d1 = 0;
	req3_d2 = 0;
	req3_data = 0 ;
	req3_r1 = 0;
	req3_tag = 0;
	req4_cmd = 0;
	req4_d1 = 0;
	req4_d2 = 0;
	req4_data = 0 ;
	req4_r1 = 0;
	req4_tag = 0; 

	#200

	  req1_cmd = 1;
	req1_d1 = 8;
	req1_d2 = 1;
	req1_data = 0 ;
	req1_r1 = 8;
	req1_tag = 3;
	req2_cmd = 1;
	req2_d1 = 8;
	req2_d2 = 1;
	req2_data = 0 ;
	req2_r1 = 8;
	req2_tag = 3;
	req3_cmd = 1;
	req3_d1 = 8;
	req3_d2 = 1;
	req3_data = 0 ;
	req3_r1 = 8;
	req3_tag = 3;
	req4_cmd = 1;
	req4_d1 = 8;
	req4_d2 = 1;
	req4_data = 0 ;
	req4_r1 = 8;
	req4_tag = 3;

	#5000
	
	#2000 $stop;

     end // initial begin

   always
     @ (reset or c_clk or req1_cmd or req1_data or req2_cmd or req2_data or req3_cmd or req3_data or req4_cmd or req4_data) begin
	
	$display ("%t: r:%b \n 
1c:%d,1d1:%d,1d2:%d,1d:%d,1r:%d,1t:%d
2c:%d,2d1:%d,2d2:%d,2d:%d,2r:%d,2t:%d
3c:%d,3d1:%d,3d2:%d,3d:%d,3r:%d,3t:%d
4c:%d,4d1:%d,4d2:%d,4d:%d,4r:%d,4t:%d  
1r:%d,1d:%d,1t:%d  
2r:%d,2d:%d,2t:%d  
3r:%d,3d:%d,3t:%d  
4r:%d,4d:%d,4t:%d \n\n", 

$time, reset, 
req1_cmd, req1_d1, req1_d2, req1_data, req1_r1, req1_tag,
req2_cmd, req2_d1, req2_d2, req2_data, req2_r1, req2_tag,
req3_cmd, req3_d1, req3_d2, req3_data, req3_r1, req3_tag,
req4_cmd, req4_d1, req4_d2, req4_data, req4_r1, req4_tag,
out1_resp, out1_data, out1_tag,
out2_resp, out2_data, out2_tag,
out3_resp, out3_data, out3_tag,
out4_resp, out4_data, out4_tag);
	
     end


endmodule // test5

module atest6and7;

   wire [0:1] out1_resp, out1_tag, out2_resp, out2_tag, out3_resp, out3_tag, out4_resp, out4_tag;
   wire [0:31] out1_data, out2_data, out3_data, out4_data;
   wire 	 scan_out;
   
   reg [0:1] req1_tag, req2_tag, req3_tag, req4_tag;
   reg       a_clk, b_clk, c_clk, reset, scan_in;
   reg [0:3] req1_cmd, req1_d1, req1_d2, req1_r1, req2_cmd, req2_d1, req2_d2, req2_r1, req3_cmd, req3_d1, req3_d2, req3_r1, req4_cmd, req4_d1, req4_d2, req4_r1;
   reg [0:31] req1_data, req2_data, req3_data, req4_data;
      
   wire [0:15] add_shift_branch_data;
   wire [0:3]  adder_cmd, adder_out_cmd;
   wire [0:4]  adder_follow_branch;
   wire [0:31] adder_out_data1, adder_out_data2, adder_out_data3, adder_out_data4;
   wire [0:1]  adder_out_resp1, adder_out_resp2, adder_out_resp3, adder_out_resp4, adder_out_tag1, adder_out_tag2, adder_out_tag3, adder_out_tag4;
   wire        adder_overflow;
   wire [0:3]  adder_read_adr1, adder_read_adr2; 
   wire [0:63] adder_read_d1, adder_read_d2; 
   wire        adder_read_valid1, adder_read_valid2;
   wire [0:63] adder_result;
   wire [0:4]  adder_result_reg;
   wire [0:3]  adder_tag, adder_write_adr;
   wire [0:31] adder_write_data;
   wire        adder_write_valid;
   wire [0:3]  hold1_cmd, hold1_d1, hold1_d2;
   wire [0:31] hold1_data;
   wire [0:3]  hold1_r1;
   wire [0:1]  hold1_tag;
   wire [0:3]  hold2_cmd, hold2_d1, hold2_d2;
   wire [0:31] hold2_data;
   wire [0:3]  hold2_r1;
   wire [0:1]  hold2_tag;
   wire [0:3]  hold3_cmd, hold3_d1, hold3_d2;
   wire [0:31] hold3_data;
   wire [0:3]  hold3_r1;
   wire [0:1]  hold3_tag;
   wire [0:3]  hold4_cmd;
   wire [0:3]  hold4_d1,  hold4_d2;
   wire [0:31] hold4_data;
   wire [0:3]  hold4_r1;
   wire [0:1]  hold4_tag;
   wire port1_invalid_op;
   wire [0:1] port1_invalid_tag;
   wire       port2_invalid_op;
   wire [0:1] port2_invalid_tag;
   wire       port3_invalid_op;
   wire [0:1] port3_invalid_tag;
   wire       port4_invalid_op;
   wire [0:1] port4_invalid_tag;
   wire [0:3] prio_adder_cmd;
   wire [0:4] prio_adder_data1, prio_adder_data2;
   wire [0:4] prio_adder_follow_branch;
   wire       prio_adder_out_vld;
   wire [0:4] prio_adder_result;
   wire [0:3] prio_adder_tag,  prio_shift_cmd;
   wire [0:31] prio_shift_data;
   wire [0:4]  prio_shift_data1,  prio_shift_data2;
   wire [0:4]  prio_shift_follow_branch;
   wire        prio_shift_out_vld;
   wire [0:4]  prio_shift_result;
   wire [0:3]  prio_shift_tag;
   wire        scan_connect1, scan_connect10, scan_connect11, scan_connect12, scan_connect2, scan_connect3, scan_connect4, scan_connect5, scan_connect6, scan_connect7, scan_connect8, scan_connect9;
   wire [0:3]  shift_cmd;
   wire [0:4]  shift_follow_branch;
   wire [0:3]  shift_out_cmd;
   wire [0:31] shift_out_data1, shift_out_data2, shift_out_data3, shift_out_data4;
   wire [0:1]  shift_out_resp1;
   wire [0:1]  shift_out_resp2, shift_out_resp3, shift_out_resp4, shift_out_tag1, shift_out_tag2, shift_out_tag3, shift_out_tag4;
   wire [0:3]  shift_read_adr1,  shift_read_adr2;
   wire [0:63] shift_read_d1, shift_read_d2;
   wire        shift_read_valid1,  shift_read_valid2;
   wire [0:63] shift_result;
   wire [0:4]  shift_result_reg;
   wire [0:3]  shift_tag, shift_write_adr;
   wire [0:31] shift_write_data;
   wire        shift_write_valid, store_data_valid;
   wire [0:63] store_val;
   
   calc3_top ( out1_data, out1_resp, out1_tag, out2_data, out2_resp, out2_tag, out3_data, out3_resp, out3_tag, out4_data, out4_resp, out4_tag, scan_out, a_clk, b_clk, c_clk, req1_cmd, req1_d1, req1_d2, req1_data, req1_r1, req1_tag, req2_cmd, req2_d1, req2_d2, req2_data, req2_r1, req2_tag, req3_cmd, req3_d1, req3_d2, req3_data, req3_r1, req3_tag, req4_cmd, req4_d1, req4_d2, req4_data, req4_r1, req4_tag, reset, scan_in);
      
   initial 
     begin
	c_clk = 0;
	a_clk = 0;
	b_clk = 0;
	scan_in = 0;
	
     end
   
	
   always #100 c_clk = ~c_clk;
   
   initial
     begin

	reset = 1;
	
	#600 

	  reset = 0;
	req1_cmd = 0;
	req1_d1 = 0;
	req1_d2 = 0;
	req1_data = 0 ;
	req1_r1 = 0;
	req1_tag = 0;
	req2_cmd = 0;
	req2_d1 = 0;
	req2_d2 = 0;
	req2_data = 0 ;
	req2_r1 = 0;
	req2_tag = 0;
	req3_cmd = 0;
	req3_d1 = 0;
	req3_d2 = 0;
	req3_data = 0 ;
	req3_r1 = 0;
	req3_tag = 0;
	req4_cmd = 0;
	req4_d1 = 0;
	req4_d2 = 0;
	req4_data = 0 ;
	req4_r1 = 0;
	req4_tag = 0;
	
	#200 

	  req1_cmd = 9;
	req1_d1 = 0;
	req1_d2 = 0;
	req1_data = 0 ;
	req1_r1 = 1;
	req1_tag = 1;
	  
	req3_cmd = 9;
	req3_d1 = 0;
	req3_d2 = 0;
	req3_data = 2 ;
	req3_r1 = 2;
	req3_tag = 0;

	req4_cmd = 9;
	req4_d1 = 0;
	req4_d2 = 0;
	req4_data = 3 ;
	req4_r1 = 3;
	req4_tag = 0;
	
	#200 

	  req1_cmd = 0;
	req1_d1 = 0;
	req1_d2 = 0;
	req1_data = 0 ;
	req1_r1 = 0;
	req1_tag = 0;
	req2_cmd = 0;
	req2_d1 = 0;
	req2_d2 = 0;
	req2_data = 0 ;
	req2_r1 = 0;
	req2_tag = 0;
	req3_cmd = 0;
	req3_d1 = 0;
	req3_d2 = 0;
	req3_data = 0 ;
	req3_r1 = 0;
	req3_tag = 0;
	req4_cmd = 0;
	req4_d1 = 0;
	req4_d2 = 0;
	req4_data = 0 ;
	req4_r1 = 0;
	req4_tag = 0;
		
	#200

	req1_cmd = 9;
	req1_d1 = 0;
	req1_d2 = 0;
	req1_data = 4 ;
	req1_r1 = 4;
	req1_tag = 1;

	req3_cmd = 9;
	req3_d1 = 0;
	req3_d2 = 0;
	req3_data = 5 ;
	req3_r1 = 5;
	req3_tag = 1;

	req4_cmd = 9;
	req4_d1 = 0;
	req4_d2 = 0;
	req4_data = 6 ;
	req4_r1 = 6;
	req4_tag = 1;

	#200

	req1_cmd = 0;
	req1_d1 = 0;
	req1_d2 = 0;
	req1_data = 0 ;
	req1_r1 = 0;
	req1_tag = 0;
	req2_cmd = 0;
	req2_d1 = 0;
	req2_d2 = 0;
	req2_data = 0 ;
	req2_r1 = 0;
	req2_tag = 0;
	req3_cmd = 0;
	req3_d1 = 0;
	req3_d2 = 0;
	req3_data = 0 ;
	req3_r1 = 0;
	req3_tag = 0;
	req4_cmd = 0;
	req4_d1 = 0;
	req4_d2 = 0;
	req4_data = 0 ;
	req4_r1 = 0;
	req4_tag = 0;

	#200

	req2_cmd = 1;
	req2_d1 = 1;
	req2_d2 = 8;
	req2_data = 0 ;
	req2_r1 = 10;
	req2_tag = 3;
	
	#2000

	req2_cmd = 0;
	req2_d1 = 0;
	req2_d2 = 0;
	req2_data = 0 ;
	req2_r1 = 0;
	req2_tag = 0;
	
	#2000 $stop;

     end // initial begin

   always
     @ (reset or c_clk or req1_cmd or req1_data or req2_cmd or req2_data or req3_cmd or req3_data or req4_cmd or req4_data) begin
	
	$display ("%t: r:%b \n 
1c:%d,1d1:%d,1d2:%d,1d:%d,1r:%d,1t:%d
2c:%d,2d1:%d,2d2:%d,2d:%d,2r:%d,2t:%d
3c:%d,3d1:%d,3d2:%d,3d:%d,3r:%d,3t:%d
4c:%d,4d1:%d,4d2:%d,4d:%d,4r:%d,4t:%d  
1r:%d,1d:%d,1t:%d  
2r:%d,2d:%d,2t:%d  
3r:%d,3d:%d,3t:%d  
4r:%d,4d:%d,4t:%d \n\n", 

$time, reset, 
req1_cmd, req1_d1, req1_d2, req1_data, req1_r1, req1_tag,
req2_cmd, req2_d1, req2_d2, req2_data, req2_r1, req2_tag,
req3_cmd, req3_d1, req3_d2, req3_data, req3_r1, req3_tag,
req4_cmd, req4_d1, req4_d2, req4_data, req4_r1, req4_tag,
out1_resp, out1_data, out1_tag,
out2_resp, out2_data, out2_tag,
out3_resp, out3_data, out3_tag,
out4_resp, out4_data, out4_tag);
	
     end


endmodule // test6and7

module atest8;

   wire [0:1] out1_resp, out1_tag, out2_resp, out2_tag, out3_resp, out3_tag, out4_resp, out4_tag;
   wire [0:31] out1_data, out2_data, out3_data, out4_data;
   wire 	 scan_out;
   
   reg [0:1] req1_tag, req2_tag, req3_tag, req4_tag;
   reg       a_clk, b_clk, c_clk, reset, scan_in;
   reg [0:3] req1_cmd, req1_d1, req1_d2, req1_r1, req2_cmd, req2_d1, req2_d2, req2_r1, req3_cmd, req3_d1, req3_d2, req3_r1, req4_cmd, req4_d1, req4_d2, req4_r1;
   reg [0:31] req1_data, req2_data, req3_data, req4_data;
      
   wire [0:15] add_shift_branch_data;
   wire [0:3]  adder_cmd, adder_out_cmd;
   wire [0:4]  adder_follow_branch;
   wire [0:31] adder_out_data1, adder_out_data2, adder_out_data3, adder_out_data4;
   wire [0:1]  adder_out_resp1, adder_out_resp2, adder_out_resp3, adder_out_resp4, adder_out_tag1, adder_out_tag2, adder_out_tag3, adder_out_tag4;
   wire        adder_overflow;
   wire [0:3]  adder_read_adr1, adder_read_adr2; 
   wire [0:63] adder_read_d1, adder_read_d2; 
   wire        adder_read_valid1, adder_read_valid2;
   wire [0:63] adder_result;
   wire [0:4]  adder_result_reg;
   wire [0:3]  adder_tag, adder_write_adr;
   wire [0:31] adder_write_data;
   wire        adder_write_valid;
   wire [0:3]  hold1_cmd, hold1_d1, hold1_d2;
   wire [0:31] hold1_data;
   wire [0:3]  hold1_r1;
   wire [0:1]  hold1_tag;
   wire [0:3]  hold2_cmd, hold2_d1, hold2_d2;
   wire [0:31] hold2_data;
   wire [0:3]  hold2_r1;
   wire [0:1]  hold2_tag;
   wire [0:3]  hold3_cmd, hold3_d1, hold3_d2;
   wire [0:31] hold3_data;
   wire [0:3]  hold3_r1;
   wire [0:1]  hold3_tag;
   wire [0:3]  hold4_cmd;
   wire [0:3]  hold4_d1,  hold4_d2;
   wire [0:31] hold4_data;
   wire [0:3]  hold4_r1;
   wire [0:1]  hold4_tag;
   wire port1_invalid_op;
   wire [0:1] port1_invalid_tag;
   wire       port2_invalid_op;
   wire [0:1] port2_invalid_tag;
   wire       port3_invalid_op;
   wire [0:1] port3_invalid_tag;
   wire       port4_invalid_op;
   wire [0:1] port4_invalid_tag;
   wire [0:3] prio_adder_cmd;
   wire [0:4] prio_adder_data1, prio_adder_data2;
   wire [0:4] prio_adder_follow_branch;
   wire       prio_adder_out_vld;
   wire [0:4] prio_adder_result;
   wire [0:3] prio_adder_tag,  prio_shift_cmd;
   wire [0:31] prio_shift_data;
   wire [0:4]  prio_shift_data1,  prio_shift_data2;
   wire [0:4]  prio_shift_follow_branch;
   wire        prio_shift_out_vld;
   wire [0:4]  prio_shift_result;
   wire [0:3]  prio_shift_tag;
   wire        scan_connect1, scan_connect10, scan_connect11, scan_connect12, scan_connect2, scan_connect3, scan_connect4, scan_connect5, scan_connect6, scan_connect7, scan_connect8, scan_connect9;
   wire [0:3]  shift_cmd;
   wire [0:4]  shift_follow_branch;
   wire [0:3]  shift_out_cmd;
   wire [0:31] shift_out_data1, shift_out_data2, shift_out_data3, shift_out_data4;
   wire [0:1]  shift_out_resp1;
   wire [0:1]  shift_out_resp2, shift_out_resp3, shift_out_resp4, shift_out_tag1, shift_out_tag2, shift_out_tag3, shift_out_tag4;
   wire [0:3]  shift_read_adr1,  shift_read_adr2;
   wire [0:63] shift_read_d1, shift_read_d2;
   wire        shift_read_valid1,  shift_read_valid2;
   wire [0:63] shift_result;
   wire [0:4]  shift_result_reg;
   wire [0:3]  shift_tag, shift_write_adr;
   wire [0:31] shift_write_data;
   wire        shift_write_valid, store_data_valid;
   wire [0:63] store_val;
   
   calc3_top ( out1_data, out1_resp, out1_tag, out2_data, out2_resp, out2_tag, out3_data, out3_resp, out3_tag, out4_data, out4_resp, out4_tag, scan_out, a_clk, b_clk, c_clk, req1_cmd, req1_d1, req1_d2, req1_data, req1_r1, req1_tag, req2_cmd, req2_d1, req2_d2, req2_data, req2_r1, req2_tag, req3_cmd, req3_d1, req3_d2, req3_data, req3_r1, req3_tag, req4_cmd, req4_d1, req4_d2, req4_data, req4_r1, req4_tag, reset, scan_in);
      
   initial 
     begin
	c_clk = 0;
	a_clk = 0;
	b_clk = 0;
	scan_in = 0;
	
     end
   
	
   always #100 c_clk = ~c_clk;
   
   initial
     begin

	reset = 1;

	
	
	#600 

	  reset = 0;
	req1_cmd = 0;
	req1_d1 = 0;
	req1_d2 = 0;
	req1_data = 0 ;
	req1_r1 = 0;
	req1_tag = 0;
	req2_cmd = 0;
	req2_d1 = 0;
	req2_d2 = 0;
	req2_data = 0 ;
	req2_r1 = 0;
	req2_tag = 0;
	req3_cmd = 0;
	req3_d1 = 0;
	req3_d2 = 0;
	req3_data = 0 ;
	req3_r1 = 0;
	req3_tag = 0;
	req4_cmd = 0;
	req4_d1 = 0;
	req4_d2 = 0;
	req4_data = 0 ;
	req4_r1 = 0;
	req4_tag = 0;
	
	#200 

	  req1_cmd = 9;
	req1_d1 = 0;
	req1_d2 = 0;
	req1_data = 1 ;
	req1_r1 = 0;
	req1_tag = 0;
	  
	req2_cmd = 9;
	req2_d1 = 0;
	req2_d2 = 0;
	req2_data = 2 ;
	req2_r1 = 1;
	req2_tag = 0;

	#200 

	  req1_cmd = 0;
	req1_d1 = 0;
	req1_d2 = 0;
	req1_data = 0 ;
	req1_r1 = 0;
	req1_tag = 0;
	req2_cmd = 0;
	req2_d1 = 0;
	req2_d2 = 0;
	req2_data = 0 ;
	req2_r1 = 0;
	req2_tag = 0;
		
	#600

	req1_cmd = 1;
	req1_d1 = 0;
	req1_d2 = 1;
	req1_data = 0 ;
	req1_r1 = 8;
	req1_tag = 1;

	req2_cmd = 5;
	req2_d1 = 0;
	req2_d2 = 1;
	req2_data = 0 ;
	req2_r1 = 8;
	req2_tag = 1;

	#200
	
	req1_cmd = 0;
	req1_d1 = 0;
	req1_d2 = 0;
	req1_data = 0 ;
	req1_r1 = 0;
	req1_tag = 0;
	req2_cmd = 0;
	req2_d1 = 0;
	req2_d2 = 0;
	req2_data = 0 ;
	req2_r1 = 0;
	req2_tag = 0;

	#600

	  req1_cmd = 1;
	req1_d1 = 0;
	req1_d2 = 1;
	req1_data = 0 ;
	req1_r1 = 10;
	req1_tag = 2;
	
	req2_cmd = 5;
	req2_d1 = 0;
	req2_d2 = 1;
	req2_data = 0 ;
	req2_r1 = 10;
	req2_tag = 2;

	#200

	  req1_cmd = 0;
	req1_d1 = 0;
	req1_d2 = 0;
	req1_data = 0 ;
	req1_r1 = 0;
	req1_tag = 0;
	req2_cmd = 0;
	req2_d1 = 0;
	req2_d2 = 0;
	req2_data = 0 ;
	req2_r1 = 0;
	req2_tag = 0;

	#200

	#2000 $stop;

     end // initial begin

   always
     @ (reset or c_clk or req1_cmd or req1_data or req2_cmd or req2_data or req3_cmd or req3_data or req4_cmd or req4_data) begin
	
	$display ("%t: r:%b \n 
1c:%d,1d1:%d,1d2:%d,1d:%d,1r:%d,1t:%d
2c:%d,2d1:%d,2d2:%d,2d:%d,2r:%d,2t:%d
3c:%d,3d1:%d,3d2:%d,3d:%d,3r:%d,3t:%d
4c:%d,4d1:%d,4d2:%d,4d:%d,4r:%d,4t:%d  
1r:%d,1d:%d,1t:%d  
2r:%d,2d:%d,2t:%d  
3r:%d,3d:%d,3t:%d  
4r:%d,4d:%d,4t:%d \n\n", 

$time, reset, 
req1_cmd, req1_d1, req1_d2, req1_data, req1_r1, req1_tag,
req2_cmd, req2_d1, req2_d2, req2_data, req2_r1, req2_tag,
req3_cmd, req3_d1, req3_d2, req3_data, req3_r1, req3_tag,
req4_cmd, req4_d1, req4_d2, req4_data, req4_r1, req4_tag,
out1_resp, out1_data, out1_tag,
out2_resp, out2_data, out2_tag,
out3_resp, out3_data, out3_tag,
out4_resp, out4_data, out4_tag);
	
     end


endmodule // test8

module atest9;

   wire [0:1] out1_resp, out1_tag, out2_resp, out2_tag, out3_resp, out3_tag, out4_resp, out4_tag;
   wire [0:31] out1_data, out2_data, out3_data, out4_data;
   wire 	 scan_out;
   
   reg [0:1] req1_tag, req2_tag, req3_tag, req4_tag;
   reg       a_clk, b_clk, c_clk, reset, scan_in;
   reg [0:3] req1_cmd, req1_d1, req1_d2, req1_r1, req2_cmd, req2_d1, req2_d2, req2_r1, req3_cmd, req3_d1, req3_d2, req3_r1, req4_cmd, req4_d1, req4_d2, req4_r1;
   reg [0:31] req1_data, req2_data, req3_data, req4_data;
      
   wire [0:15] add_shift_branch_data;
   wire [0:3]  adder_cmd, adder_out_cmd;
   wire [0:4]  adder_follow_branch;
   wire [0:31] adder_out_data1, adder_out_data2, adder_out_data3, adder_out_data4;
   wire [0:1]  adder_out_resp1, adder_out_resp2, adder_out_resp3, adder_out_resp4, adder_out_tag1, adder_out_tag2, adder_out_tag3, adder_out_tag4;
   wire        adder_overflow;
   wire [0:3]  adder_read_adr1, adder_read_adr2; 
   wire [0:63] adder_read_d1, adder_read_d2; 
   wire        adder_read_valid1, adder_read_valid2;
   wire [0:63] adder_result;
   wire [0:4]  adder_result_reg;
   wire [0:3]  adder_tag, adder_write_adr;
   wire [0:31] adder_write_data;
   wire        adder_write_valid;
   wire [0:3]  hold1_cmd, hold1_d1, hold1_d2;
   wire [0:31] hold1_data;
   wire [0:3]  hold1_r1;
   wire [0:1]  hold1_tag;
   wire [0:3]  hold2_cmd, hold2_d1, hold2_d2;
   wire [0:31] hold2_data;
   wire [0:3]  hold2_r1;
   wire [0:1]  hold2_tag;
   wire [0:3]  hold3_cmd, hold3_d1, hold3_d2;
   wire [0:31] hold3_data;
   wire [0:3]  hold3_r1;
   wire [0:1]  hold3_tag;
   wire [0:3]  hold4_cmd;
   wire [0:3]  hold4_d1,  hold4_d2;
   wire [0:31] hold4_data;
   wire [0:3]  hold4_r1;
   wire [0:1]  hold4_tag;
   wire port1_invalid_op;
   wire [0:1] port1_invalid_tag;
   wire       port2_invalid_op;
   wire [0:1] port2_invalid_tag;
   wire       port3_invalid_op;
   wire [0:1] port3_invalid_tag;
   wire       port4_invalid_op;
   wire [0:1] port4_invalid_tag;
   wire [0:3] prio_adder_cmd;
   wire [0:4] prio_adder_data1, prio_adder_data2;
   wire [0:4] prio_adder_follow_branch;
   wire       prio_adder_out_vld;
   wire [0:4] prio_adder_result;
   wire [0:3] prio_adder_tag,  prio_shift_cmd;
   wire [0:31] prio_shift_data;
   wire [0:4]  prio_shift_data1,  prio_shift_data2;
   wire [0:4]  prio_shift_follow_branch;
   wire        prio_shift_out_vld;
   wire [0:4]  prio_shift_result;
   wire [0:3]  prio_shift_tag;
   wire        scan_connect1, scan_connect10, scan_connect11, scan_connect12, scan_connect2, scan_connect3, scan_connect4, scan_connect5, scan_connect6, scan_connect7, scan_connect8, scan_connect9;
   wire [0:3]  shift_cmd;
   wire [0:4]  shift_follow_branch;
   wire [0:3]  shift_out_cmd;
   wire [0:31] shift_out_data1, shift_out_data2, shift_out_data3, shift_out_data4;
   wire [0:1]  shift_out_resp1;
   wire [0:1]  shift_out_resp2, shift_out_resp3, shift_out_resp4, shift_out_tag1, shift_out_tag2, shift_out_tag3, shift_out_tag4;
   wire [0:3]  shift_read_adr1,  shift_read_adr2;
   wire [0:63] shift_read_d1, shift_read_d2;
   wire        shift_read_valid1,  shift_read_valid2;
   wire [0:63] shift_result;
   wire [0:4]  shift_result_reg;
   wire [0:3]  shift_tag, shift_write_adr;
   wire [0:31] shift_write_data;
   wire        shift_write_valid, store_data_valid;
   wire [0:63] store_val;
   
   calc3_top ( out1_data, out1_resp, out1_tag, out2_data, out2_resp, out2_tag, out3_data, out3_resp, out3_tag, out4_data, out4_resp, out4_tag, scan_out, a_clk, b_clk, c_clk, req1_cmd, req1_d1, req1_d2, req1_data, req1_r1, req1_tag, req2_cmd, req2_d1, req2_d2, req2_data, req2_r1, req2_tag, req3_cmd, req3_d1, req3_d2, req3_data, req3_r1, req3_tag, req4_cmd, req4_d1, req4_d2, req4_data, req4_r1, req4_tag, reset, scan_in);
      
   initial 
     begin
	c_clk = 0;
	a_clk = 0;
	b_clk = 0;
	scan_in = 0;
	
     end
   
	
   always #100 c_clk = ~c_clk;
   
   initial
     begin

	reset = 1;

	
	
	#600 

	  reset = 0;
	req1_cmd = 0;
	req1_d1 = 0;
	req1_d2 = 0;
	req1_data = 0 ;
	req1_r1 = 0;
	req1_tag = 0;
	req2_cmd = 0;
	req2_d1 = 0;
	req2_d2 = 0;
	req2_data = 0 ;
	req2_r1 = 0;
	req2_tag = 0;
	req3_cmd = 0;
	req3_d1 = 0;
	req3_d2 = 0;
	req3_data = 0 ;
	req3_r1 = 0;
	req3_tag = 0;
	req4_cmd = 0;
	req4_d1 = 0;
	req4_d2 = 0;
	req4_data = 0 ;
	req4_r1 = 0;
	req4_tag = 0;
	
	#200 

	  req1_cmd = 9;
	req1_d1 = 0;
	req1_d2 = 0;
	req1_data = 0 ;
	req1_r1 = 0;
	req1_tag = 0;
	  
	req2_cmd = 9;
	req2_d1 = 0;
	req2_d2 = 0;
	req2_data = 1 ;
	req2_r1 = 1;
	req2_tag = 0;

	req3_cmd = 9;
	req3_d1 = 0;
	req3_d2 = 0;
	req3_data = 2 ;
	req3_r1 = 2;
	req3_tag = 0;

	req4_cmd = 9;
	req4_d1 = 0;
	req4_d2 = 0;
	req4_data = 3 ;
	req4_r1 = 3;
	req4_tag = 0;
	
	#200 

	  req1_cmd = 0;
	req1_d1 = 0;
	req1_d2 = 0;
	req1_data = 0 ;
	req1_r1 = 0;
	req1_tag = 0;
	req2_cmd = 0;
	req2_d1 = 0;
	req2_d2 = 0;
	req2_data = 0 ;
	req2_r1 = 0;
	req2_tag = 0;
	req3_cmd = 0;
	req3_d1 = 0;
	req3_d2 = 0;
	req3_data = 0 ;
	req3_r1 = 0;
	req3_tag = 0;
	req4_cmd = 0;
	req4_d1 = 0;
	req4_d2 = 0;
	req4_data = 0 ;
	req4_r1 = 0;
	req4_tag = 0;
		
	#200

	req1_cmd = 9;
	req1_d1 = 0;
	req1_d2 = 0;
	req1_data = 4 ;
	req1_r1 = 4;
	req1_tag = 1;

	req2_cmd = 9;
	req2_d1 = 0;
	req2_d2 = 0;
	req2_data = 5 ;
	req2_r1 = 5;
	req2_tag = 1;

	req3_cmd = 9;
	req3_d1 = 0;
	req3_d2 = 0;
	req3_data = 6 ;
	req3_r1 = 6;
	req3_tag = 1;

	req4_cmd = 9;
	req4_d1 = 0;
	req4_d2 = 0;
	req4_data = 7 ;
	req4_r1 = 7;
	req4_tag = 1;
		
	#200

	req1_cmd = 0;
	req1_d1 = 0;
	req1_d2 = 0;
	req1_data = 0 ;
	req1_r1 = 0;
	req1_tag = 0;
	req2_cmd = 0;
	req2_d1 = 0;
	req2_d2 = 0;
	req2_data = 0 ;
	req2_r1 = 0;
	req2_tag = 0;
	req3_cmd = 0;
	req3_d1 = 0;
	req3_d2 = 0;
	req3_data = 0 ;
	req3_r1 = 0;
	req3_tag = 0;
	req4_cmd = 0;
	req4_d1 = 0;
	req4_d2 = 0;
	req4_data = 0 ;
	req4_r1 = 0;
	req4_tag = 0;

	#200

	req1_cmd = 9;
	req1_d1 = 0;
	req1_d2 = 0;
	req1_data = 8 ;
	req1_r1 = 8;
	req1_tag = 2;

	req2_cmd = 9;
	req2_d1 = 0;
	req2_d2 = 0;
	req2_data = 9 ;
	req2_r1 = 9;
	req2_tag = 2;

	req3_cmd = 9;
	req3_d1 = 0;
	req3_d2 = 0;
	req3_data = 10 ;
	req3_r1 = 10;
	req3_tag = 2;

	req4_cmd = 9;
	req4_d1 = 0;
	req4_d2 = 0;
	req4_data = 11 ;
	req4_r1 = 11;
	req4_tag = 2;
		
	#2000

	  req1_cmd = 0;
	req1_d1 = 0;
	req1_d2 = 0;
	req1_data = 0 ;
	req1_r1 = 0;
	req1_tag = 0;
	req2_cmd = 0;
	req2_d1 = 0;
	req2_d2 = 0;
	req2_data = 0 ;
	req2_r1 = 0;
	req2_tag = 0;
	req3_cmd = 0;
	req3_d1 = 0;
	req3_d2 = 0;
	req3_data = 0 ;
	req3_r1 = 0;
	req3_tag = 0;
	req4_cmd = 0;
	req4_d1 = 0;
	req4_d2 = 0;
	req4_data = 0 ;
	req4_r1 = 0;
	req4_tag = 0; 

	#200

	req1_cmd = 12;
	req1_d1 = 0;
	req1_d2 = 0;
	req1_data = 0;
	req1_r1 = 0;
	req1_tag = 3;

	#200

	  req1_cmd = 0;
	req1_d1 = 0;
	req1_d2 = 0;
	req1_data = 0 ;
	req1_r1 = 0;
	req1_tag = 0;

	#200

	  req1_cmd = 9;
	req1_d1 = 0;
	req1_d2 = 0;
	req1_data = {32{1'b1}} ;
	req1_r1 = 15;
	req1_tag = 0;
	

	#200;
	req1_cmd = 0;
	req1_d1 = 0;
	req1_d2 = 0;
	req1_data = 0 ;
	req1_r1 = 0;
	req1_tag = 0;

	#600

	  req1_cmd = 12;
	req1_d1 = 1;
	req1_d2 = 0;
	req1_data = 0 ;
	req1_r1 = 0;
	req1_tag = 3;

	#200;
	req1_cmd = 0;
	req1_d1 = 0;
	req1_d2 = 0;
	req1_data = 0 ;
	req1_r1 = 0;
	req1_tag = 0;
	
	#2000 $stop;

     end // initial begin

   always
     @ (reset or c_clk or req1_cmd or req1_data or req2_cmd or req2_data or req3_cmd or req3_data or req4_cmd or req4_data) begin
	
	$display ("%t: r:%b \n 
1c:%d,1d1:%d,1d2:%d,1d:%d,1r:%d,1t:%d
2c:%d,2d1:%d,2d2:%d,2d:%d,2r:%d,2t:%d
3c:%d,3d1:%d,3d2:%d,3d:%d,3r:%d,3t:%d
4c:%d,4d1:%d,4d2:%d,4d:%d,4r:%d,4t:%d  
1r:%d,1d:%d,1t:%d  
2r:%d,2d:%d,2t:%d  
3r:%d,3d:%d,3t:%d  
4r:%d,4d:%d,4t:%d \n\n", 

$time, reset, 
req1_cmd, req1_d1, req1_d2, req1_data, req1_r1, req1_tag,
req2_cmd, req2_d1, req2_d2, req2_data, req2_r1, req2_tag,
req3_cmd, req3_d1, req3_d2, req3_data, req3_r1, req3_tag,
req4_cmd, req4_d1, req4_d2, req4_data, req4_r1, req4_tag,
out1_resp, out1_data, out1_tag,
out2_resp, out2_data, out2_tag,
out3_resp, out3_data, out3_tag,
out4_resp, out4_data, out4_tag);
	
     end


endmodule // test9


module atest10;

   wire [0:1] out1_resp, out1_tag, out2_resp, out2_tag, out3_resp, out3_tag, out4_resp, out4_tag;
   wire [0:31] out1_data, out2_data, out3_data, out4_data;
   wire 	 scan_out;
   
   reg [0:1] req1_tag, req2_tag, req3_tag, req4_tag;
   reg       a_clk, b_clk, c_clk, reset, scan_in;
   reg [0:3] req1_cmd, req1_d1, req1_d2, req1_r1, req2_cmd, req2_d1, req2_d2, req2_r1, req3_cmd, req3_d1, req3_d2, req3_r1, req4_cmd, req4_d1, req4_d2, req4_r1;
   reg [0:31] req1_data, req2_data, req3_data, req4_data;
      
   wire [0:15] add_shift_branch_data;
   wire [0:3]  adder_cmd, adder_out_cmd;
   wire [0:4]  adder_follow_branch;
   wire [0:31] adder_out_data1, adder_out_data2, adder_out_data3, adder_out_data4;
   wire [0:1]  adder_out_resp1, adder_out_resp2, adder_out_resp3, adder_out_resp4, adder_out_tag1, adder_out_tag2, adder_out_tag3, adder_out_tag4;
   wire        adder_overflow;
   wire [0:3]  adder_read_adr1, adder_read_adr2; 
   wire [0:63] adder_read_d1, adder_read_d2; 
   wire        adder_read_valid1, adder_read_valid2;
   wire [0:63] adder_result;
   wire [0:4]  adder_result_reg;
   wire [0:3]  adder_tag, adder_write_adr;
   wire [0:31] adder_write_data;
   wire        adder_write_valid;
   wire [0:3]  hold1_cmd, hold1_d1, hold1_d2;
   wire [0:31] hold1_data;
   wire [0:3]  hold1_r1;
   wire [0:1]  hold1_tag;
   wire [0:3]  hold2_cmd, hold2_d1, hold2_d2;
   wire [0:31] hold2_data;
   wire [0:3]  hold2_r1;
   wire [0:1]  hold2_tag;
   wire [0:3]  hold3_cmd, hold3_d1, hold3_d2;
   wire [0:31] hold3_data;
   wire [0:3]  hold3_r1;
   wire [0:1]  hold3_tag;
   wire [0:3]  hold4_cmd;
   wire [0:3]  hold4_d1,  hold4_d2;
   wire [0:31] hold4_data;
   wire [0:3]  hold4_r1;
   wire [0:1]  hold4_tag;
   wire port1_invalid_op;
   wire [0:1] port1_invalid_tag;
   wire       port2_invalid_op;
   wire [0:1] port2_invalid_tag;
   wire       port3_invalid_op;
   wire [0:1] port3_invalid_tag;
   wire       port4_invalid_op;
   wire [0:1] port4_invalid_tag;
   wire [0:3] prio_adder_cmd;
   wire [0:4] prio_adder_data1, prio_adder_data2;
   wire [0:4] prio_adder_follow_branch;
   wire       prio_adder_out_vld;
   wire [0:4] prio_adder_result;
   wire [0:3] prio_adder_tag,  prio_shift_cmd;
   wire [0:31] prio_shift_data;
   wire [0:4]  prio_shift_data1,  prio_shift_data2;
   wire [0:4]  prio_shift_follow_branch;
   wire        prio_shift_out_vld;
   wire [0:4]  prio_shift_result;
   wire [0:3]  prio_shift_tag;
   wire        scan_connect1, scan_connect10, scan_connect11, scan_connect12, scan_connect2, scan_connect3, scan_connect4, scan_connect5, scan_connect6, scan_connect7, scan_connect8, scan_connect9;
   wire [0:3]  shift_cmd;
   wire [0:4]  shift_follow_branch;
   wire [0:3]  shift_out_cmd;
   wire [0:31] shift_out_data1, shift_out_data2, shift_out_data3, shift_out_data4;
   wire [0:1]  shift_out_resp1;
   wire [0:1]  shift_out_resp2, shift_out_resp3, shift_out_resp4, shift_out_tag1, shift_out_tag2, shift_out_tag3, shift_out_tag4;
   wire [0:3]  shift_read_adr1,  shift_read_adr2;
   wire [0:63] shift_read_d1, shift_read_d2;
   wire        shift_read_valid1,  shift_read_valid2;
   wire [0:63] shift_result;
   wire [0:4]  shift_result_reg;
   wire [0:3]  shift_tag, shift_write_adr;
   wire [0:31] shift_write_data;
   wire        shift_write_valid, store_data_valid;
   wire [0:63] store_val;
   
   calc3_top ( out1_data, out1_resp, out1_tag, out2_data, out2_resp, out2_tag, out3_data, out3_resp, out3_tag, out4_data, out4_resp, out4_tag, scan_out, a_clk, b_clk, c_clk, req1_cmd, req1_d1, req1_d2, req1_data, req1_r1, req1_tag, req2_cmd, req2_d1, req2_d2, req2_data, req2_r1, req2_tag, req3_cmd, req3_d1, req3_d2, req3_data, req3_r1, req3_tag, req4_cmd, req4_d1, req4_d2, req4_data, req4_r1, req4_tag, reset, scan_in);
      
   initial 
     begin
	c_clk = 0;
	a_clk = 0;
	b_clk = 0;
	scan_in = 0;
	
     end
   
	
   always #100 c_clk = ~c_clk;
   
   initial
     begin

	reset = 1;

	#600 

	  reset = 0;
	req1_cmd = 0;
	req1_d1 = 0;
	req1_d2 = 0;
	req1_data = 0 ;
	req1_r1 = 0;
	req1_tag = 0;
	req2_cmd = 0;
	req2_d1 = 0;
	req2_d2 = 0;
	req2_data = 0 ;
	req2_r1 = 0;
	req2_tag = 0;
	req3_cmd = 0;
	req3_d1 = 0;
	req3_d2 = 0;
	req3_data = 0 ;
	req3_r1 = 0;
	req3_tag = 0;
	req4_cmd = 0;
	req4_d1 = 0;
	req4_d2 = 0;
	req4_data = 0 ;
	req4_r1 = 0;
	req4_tag = 0;
	
	#200 

	  req1_cmd = 9;
	req1_d1 = 0;
	req1_d2 = 0;
	req1_data = 1 ;
	req1_r1 = 0;
	req1_tag = 0;
	  
	req2_cmd = 9;
	req2_d1 = 0;
	req2_d2 = 0;
	req2_data = 2 ;
	req2_r1 = 1;
	req2_tag = 0;

	req3_cmd = 9;
	req3_d1 = 0;
	req3_d2 = 0;
	req3_data = 2 ;
	req3_r1 = 2;
	req3_tag = 0;

	req4_cmd = 9;
	req4_d1 = 0;
	req4_d2 = 0;
	req4_data = 3 ;
	req4_r1 = 3;
	req4_tag = 0;
	
	#200 

	  req1_cmd = 0;
	req1_d1 = 0;
	req1_d2 = 0;
	req1_data = 0 ;
	req1_r1 = 0;
	req1_tag = 0;
	req2_cmd = 0;
	req2_d1 = 0;
	req2_d2 = 0;
	req2_data = 0 ;
	req2_r1 = 0;
	req2_tag = 0;
		
	#600

	req1_cmd = 1;
	req1_d1 = 0;
	req1_d2 = 1;
	req1_data = 0 ;
	req1_r1 = 8;
	req1_tag = 1;

	req2_cmd = 5;
	req2_d1 = 0;
	req2_d2 = 1;
	req2_data = 0 ;
	req2_r1 = 8;
	req2_tag = 1;

	req3_cmd = 1;
	req3_d1 = 0;
	req3_d2 = 1;
	req3_data = 0 ;
	req3_r1 = 8;
	req3_tag = 1;

	#200

	req1_cmd = 0;
	req1_d1 = 0;
	req1_d2 = 0;
	req1_data = 0 ;
	req1_r1 = 0;
	req1_tag = 0;
	req2_cmd = 0;
	req2_d1 = 0;
	req2_d2 = 0;
	req2_data = 0 ;
	req2_r1 = 0;
	req2_tag = 0;
	req3_cmd = 0;
	req3_d1 = 0;
	req3_d2 = 0;
	req3_data = 0 ;
	req3_r1 = 0;
	req3_tag = 0;

	#200

	  req1_cmd = 1;
	req1_d1 = 0;
	req1_d2 = 1;
	req1_data = 0 ;
	req1_r1 = 8;
	req1_tag = 2;
	
	req2_cmd = 1;
	req2_d1 = 0;
	req2_d2 = 1;
	req2_data = 0 ;
	req2_r1 = 12;
	req2_tag = 2;
	
	req3_cmd = 1;
	req3_d1 = 0;
	req3_d2 = 1;
	req3_data = 0 ;
	req3_r1 = 8;
	req3_tag = 3;
	
	#200

	  req1_cmd = 0;
	req1_d1 = 0;
	req1_d2 = 0;
	req1_data = 0 ;
	req1_r1 = 0;
	req1_tag = 0;
	req2_cmd = 0;
	req2_d1 = 0;
	req2_d2 = 0;
	req2_data = 0 ;
	req2_r1 = 0;
	req2_tag = 0;
	req3_cmd = 0;
	req3_d1 = 0;
	req3_d2 = 0;
	req3_data = 0 ;
	req3_r1 = 0;
	req3_tag = 0;
	req4_cmd = 0;
	req4_d1 = 0;
	req4_d2 = 0;
	req4_data = 0 ;
	req4_r1 = 0;
	req4_tag = 0; 

	#2000 $stop;

     end // initial begin

   always
     @ (reset or c_clk or req1_cmd or req1_data or req2_cmd or req2_data or req3_cmd or req3_data or req4_cmd or req4_data) begin
	
	$display ("%t: r:%b \n 
1c:%d,1d1:%d,1d2:%d,1d:%d,1r:%d,1t:%d
2c:%d,2d1:%d,2d2:%d,2d:%d,2r:%d,2t:%d
3c:%d,3d1:%d,3d2:%d,3d:%d,3r:%d,3t:%d
4c:%d,4d1:%d,4d2:%d,4d:%d,4r:%d,4t:%d  
1r:%d,1d:%d,1t:%d  
2r:%d,2d:%d,2t:%d  
3r:%d,3d:%d,3t:%d  
4r:%d,4d:%d,4t:%d \n\n", 

$time, reset, 
req1_cmd, req1_d1, req1_d2, req1_data, req1_r1, req1_tag,
req2_cmd, req2_d1, req2_d2, req2_data, req2_r1, req2_tag,
req3_cmd, req3_d1, req3_d2, req3_data, req3_r1, req3_tag,
req4_cmd, req4_d1, req4_d2, req4_data, req4_r1, req4_tag,
out1_resp, out1_data, out1_tag,
out2_resp, out2_data, out2_tag,
out3_resp, out3_data, out3_tag,
out4_resp, out4_data, out4_tag);
	
     end


endmodule // test10

module atest11;

   wire [0:1] out1_resp, out1_tag, out2_resp, out2_tag, out3_resp, out3_tag, out4_resp, out4_tag;
   wire [0:31] out1_data, out2_data, out3_data, out4_data;
   wire 	 scan_out;
   
   reg [0:1] req1_tag, req2_tag, req3_tag, req4_tag;
   reg       a_clk, b_clk, c_clk, reset, scan_in;
   reg [0:3] req1_cmd, req1_d1, req1_d2, req1_r1, req2_cmd, req2_d1, req2_d2, req2_r1, req3_cmd, req3_d1, req3_d2, req3_r1, req4_cmd, req4_d1, req4_d2, req4_r1;
   reg [0:31] req1_data, req2_data, req3_data, req4_data;
      
   wire [0:15] add_shift_branch_data;
   wire [0:3]  adder_cmd, adder_out_cmd;
   wire [0:4]  adder_follow_branch;
   wire [0:31] adder_out_data1, adder_out_data2, adder_out_data3, adder_out_data4;
   wire [0:1]  adder_out_resp1, adder_out_resp2, adder_out_resp3, adder_out_resp4, adder_out_tag1, adder_out_tag2, adder_out_tag3, adder_out_tag4;
   wire        adder_overflow;
   wire [0:3]  adder_read_adr1, adder_read_adr2; 
   wire [0:63] adder_read_d1, adder_read_d2; 
   wire        adder_read_valid1, adder_read_valid2;
   wire [0:63] adder_result;
   wire [0:4]  adder_result_reg;
   wire [0:3]  adder_tag, adder_write_adr;
   wire [0:31] adder_write_data;
   wire        adder_write_valid;
   wire [0:3]  hold1_cmd, hold1_d1, hold1_d2;
   wire [0:31] hold1_data;
   wire [0:3]  hold1_r1;
   wire [0:1]  hold1_tag;
   wire [0:3]  hold2_cmd, hold2_d1, hold2_d2;
   wire [0:31] hold2_data;
   wire [0:3]  hold2_r1;
   wire [0:1]  hold2_tag;
   wire [0:3]  hold3_cmd, hold3_d1, hold3_d2;
   wire [0:31] hold3_data;
   wire [0:3]  hold3_r1;
   wire [0:1]  hold3_tag;
   wire [0:3]  hold4_cmd;
   wire [0:3]  hold4_d1,  hold4_d2;
   wire [0:31] hold4_data;
   wire [0:3]  hold4_r1;
   wire [0:1]  hold4_tag;
   wire port1_invalid_op;
   wire [0:1] port1_invalid_tag;
   wire       port2_invalid_op;
   wire [0:1] port2_invalid_tag;
   wire       port3_invalid_op;
   wire [0:1] port3_invalid_tag;
   wire       port4_invalid_op;
   wire [0:1] port4_invalid_tag;
   wire [0:3] prio_adder_cmd;
   wire [0:4] prio_adder_data1, prio_adder_data2;
   wire [0:4] prio_adder_follow_branch;
   wire       prio_adder_out_vld;
   wire [0:4] prio_adder_result;
   wire [0:3] prio_adder_tag,  prio_shift_cmd;
   wire [0:31] prio_shift_data;
   wire [0:4]  prio_shift_data1,  prio_shift_data2;
   wire [0:4]  prio_shift_follow_branch;
   wire        prio_shift_out_vld;
   wire [0:4]  prio_shift_result;
   wire [0:3]  prio_shift_tag;
   wire        scan_connect1, scan_connect10, scan_connect11, scan_connect12, scan_connect2, scan_connect3, scan_connect4, scan_connect5, scan_connect6, scan_connect7, scan_connect8, scan_connect9;
   wire [0:3]  shift_cmd;
   wire [0:4]  shift_follow_branch;
   wire [0:3]  shift_out_cmd;
   wire [0:31] shift_out_data1, shift_out_data2, shift_out_data3, shift_out_data4;
   wire [0:1]  shift_out_resp1;
   wire [0:1]  shift_out_resp2, shift_out_resp3, shift_out_resp4, shift_out_tag1, shift_out_tag2, shift_out_tag3, shift_out_tag4;
   wire [0:3]  shift_read_adr1,  shift_read_adr2;
   wire [0:63] shift_read_d1, shift_read_d2;
   wire        shift_read_valid1,  shift_read_valid2;
   wire [0:63] shift_result;
   wire [0:4]  shift_result_reg;
   wire [0:3]  shift_tag, shift_write_adr;
   wire [0:31] shift_write_data;
   wire        shift_write_valid, store_data_valid;
   wire [0:63] store_val;
   
   calc3_top ( out1_data, out1_resp, out1_tag, out2_data, out2_resp, out2_tag, out3_data, out3_resp, out3_tag, out4_data, out4_resp, out4_tag, scan_out, a_clk, b_clk, c_clk, req1_cmd, req1_d1, req1_d2, req1_data, req1_r1, req1_tag, req2_cmd, req2_d1, req2_d2, req2_data, req2_r1, req2_tag, req3_cmd, req3_d1, req3_d2, req3_data, req3_r1, req3_tag, req4_cmd, req4_d1, req4_d2, req4_data, req4_r1, req4_tag, reset, scan_in);
      
   initial 
     begin
	c_clk = 0;
	a_clk = 0;
	b_clk = 0;
	scan_in = 0;
	
     end
   
	
   always #100 c_clk = ~c_clk;
   
   initial
     begin

	reset = 1;

	
	
	#600 

	  reset = 0;
	req1_cmd = 0;
	req1_d1 = 0;
	req1_d2 = 0;
	req1_data = 0 ;
	req1_r1 = 0;
	req1_tag = 0;
	req2_cmd = 0;
	req2_d1 = 0;
	req2_d2 = 0;
	req2_data = 0 ;
	req2_r1 = 0;
	req2_tag = 0;
	req3_cmd = 0;
	req3_d1 = 0;
	req3_d2 = 0;
	req3_data = 0 ;
	req3_r1 = 0;
	req3_tag = 0;
	req4_cmd = 0;
	req4_d1 = 0;
	req4_d2 = 0;
	req4_data = 0 ;
	req4_r1 = 0;
	req4_tag = 0;
	
	#200 

	  req1_cmd = 9;
	req1_d1 = 0;
	req1_d2 = 0;
	req1_data = 1 ;
	req1_r1 = 1;
	req1_tag = 0;

	req3_d1 = 9;
	req3_d2 = 0;
	req3_data = 2 ;
	req3_r1 = 2;
	req3_tag = 0;

	
	#200 

	  req1_cmd = 9;
	req1_d1 = 0;
	req1_d2 = 0;
	req1_data = 3 ;
	req1_r1 = 3;
	req1_tag = 1;

	req2_cmd =9;
	req2_d1 = 0;
	req2_d2 = 0;
	req2_data = {32{1'b1}} ;
	req2_r1 = 4;
	req2_tag = 0;

	req3_cmd = 0;
	req3_d1 = 0;
	req3_d2 = 0;
	req3_data = 0 ;
	req3_r1 = 0;
	req3_tag = 0;
		
	#200

	req1_cmd = 0;
	req1_d1 = 0;
	req1_d2 = 0;
	req1_data = 0 ;
	req1_r1 = 0;
	req1_tag = 0;

	req2_cmd = 1;
	req2_d1 = 0;
	req2_d2 = 1;
	req2_data = 0 ;
	req2_r1 = 4;
	req2_tag = 3;
		
	#200

	req2_cmd = 0;
	req2_d1 = 0;
	req2_d2 = 0;
	req2_data = 0 ;
	req2_r1 = 0;
	req2_tag = 0;

	#2000 $stop;

     end // initial begin

   always
     @ (reset or c_clk or req1_cmd or req1_data or req2_cmd or req2_data or req3_cmd or req3_data or req4_cmd or req4_data) begin
	
	$display ("%t: r:%b \n 
1c:%d,1d1:%d,1d2:%d,1d:%d,1r:%d,1t:%d
2c:%d,2d1:%d,2d2:%d,2d:%d,2r:%d,2t:%d
3c:%d,3d1:%d,3d2:%d,3d:%d,3r:%d,3t:%d
4c:%d,4d1:%d,4d2:%d,4d:%d,4r:%d,4t:%d  
1r:%d,1d:%d,1t:%d  
2r:%d,2d:%d,2t:%d  
3r:%d,3d:%d,3t:%d  
4r:%d,4d:%d,4t:%d \n\n", 

$time, reset, 
req1_cmd, req1_d1, req1_d2, req1_data, req1_r1, req1_tag,
req2_cmd, req2_d1, req2_d2, req2_data, req2_r1, req2_tag,
req3_cmd, req3_d1, req3_d2, req3_data, req3_r1, req3_tag,
req4_cmd, req4_d1, req4_d2, req4_data, req4_r1, req4_tag,
out1_resp, out1_data, out1_tag,
out2_resp, out2_data, out2_tag,
out3_resp, out3_data, out3_tag,
out4_resp, out4_data, out4_tag);
	
     end


endmodule // test11

module atest12;

   wire [0:1] out1_resp, out1_tag, out2_resp, out2_tag, out3_resp, out3_tag, out4_resp, out4_tag;
   wire [0:31] out1_data, out2_data, out3_data, out4_data;
   wire 	 scan_out;
   
   reg [0:1] req1_tag, req2_tag, req3_tag, req4_tag;
   reg       a_clk, b_clk, c_clk, reset, scan_in;
   reg [0:3] req1_cmd, req1_d1, req1_d2, req1_r1, req2_cmd, req2_d1, req2_d2, req2_r1, req3_cmd, req3_d1, req3_d2, req3_r1, req4_cmd, req4_d1, req4_d2, req4_r1;
   reg [0:31] req1_data, req2_data, req3_data, req4_data;
      
   wire [0:15] add_shift_branch_data;
   wire [0:3]  adder_cmd, adder_out_cmd;
   wire [0:4]  adder_follow_branch;
   wire [0:31] adder_out_data1, adder_out_data2, adder_out_data3, adder_out_data4;
   wire [0:1]  adder_out_resp1, adder_out_resp2, adder_out_resp3, adder_out_resp4, adder_out_tag1, adder_out_tag2, adder_out_tag3, adder_out_tag4;
   wire        adder_overflow;
   wire [0:3]  adder_read_adr1, adder_read_adr2; 
   wire [0:63] adder_read_d1, adder_read_d2; 
   wire        adder_read_valid1, adder_read_valid2;
   wire [0:63] adder_result;
   wire [0:4]  adder_result_reg;
   wire [0:3]  adder_tag, adder_write_adr;
   wire [0:31] adder_write_data;
   wire        adder_write_valid;
   wire [0:3]  hold1_cmd, hold1_d1, hold1_d2;
   wire [0:31] hold1_data;
   wire [0:3]  hold1_r1;
   wire [0:1]  hold1_tag;
   wire [0:3]  hold2_cmd, hold2_d1, hold2_d2;
   wire [0:31] hold2_data;
   wire [0:3]  hold2_r1;
   wire [0:1]  hold2_tag;
   wire [0:3]  hold3_cmd, hold3_d1, hold3_d2;
   wire [0:31] hold3_data;
   wire [0:3]  hold3_r1;
   wire [0:1]  hold3_tag;
   wire [0:3]  hold4_cmd;
   wire [0:3]  hold4_d1,  hold4_d2;
   wire [0:31] hold4_data;
   wire [0:3]  hold4_r1;
   wire [0:1]  hold4_tag;
   wire port1_invalid_op;
   wire [0:1] port1_invalid_tag;
   wire       port2_invalid_op;
   wire [0:1] port2_invalid_tag;
   wire       port3_invalid_op;
   wire [0:1] port3_invalid_tag;
   wire       port4_invalid_op;
   wire [0:1] port4_invalid_tag;
   wire [0:3] prio_adder_cmd;
   wire [0:4] prio_adder_data1, prio_adder_data2;
   wire [0:4] prio_adder_follow_branch;
   wire       prio_adder_out_vld;
   wire [0:4] prio_adder_result;
   wire [0:3] prio_adder_tag,  prio_shift_cmd;
   wire [0:31] prio_shift_data;
   wire [0:4]  prio_shift_data1,  prio_shift_data2;
   wire [0:4]  prio_shift_follow_branch;
   wire        prio_shift_out_vld;
   wire [0:4]  prio_shift_result;
   wire [0:3]  prio_shift_tag;
   wire        scan_connect1, scan_connect10, scan_connect11, scan_connect12, scan_connect2, scan_connect3, scan_connect4, scan_connect5, scan_connect6, scan_connect7, scan_connect8, scan_connect9;
   wire [0:3]  shift_cmd;
   wire [0:4]  shift_follow_branch;
   wire [0:3]  shift_out_cmd;
   wire [0:31] shift_out_data1, shift_out_data2, shift_out_data3, shift_out_data4;
   wire [0:1]  shift_out_resp1;
   wire [0:1]  shift_out_resp2, shift_out_resp3, shift_out_resp4, shift_out_tag1, shift_out_tag2, shift_out_tag3, shift_out_tag4;
   wire [0:3]  shift_read_adr1,  shift_read_adr2;
   wire [0:63] shift_read_d1, shift_read_d2;
   wire        shift_read_valid1,  shift_read_valid2;
   wire [0:63] shift_result;
   wire [0:4]  shift_result_reg;
   wire [0:3]  shift_tag, shift_write_adr;
   wire [0:31] shift_write_data;
   wire        shift_write_valid, store_data_valid;
   wire [0:63] store_val;
   
   calc3_top ( out1_data, out1_resp, out1_tag, out2_data, out2_resp, out2_tag, out3_data, out3_resp, out3_tag, out4_data, out4_resp, out4_tag, scan_out, a_clk, b_clk, c_clk, req1_cmd, req1_d1, req1_d2, req1_data, req1_r1, req1_tag, req2_cmd, req2_d1, req2_d2, req2_data, req2_r1, req2_tag, req3_cmd, req3_d1, req3_d2, req3_data, req3_r1, req3_tag, req4_cmd, req4_d1, req4_d2, req4_data, req4_r1, req4_tag, reset, scan_in);
      
   initial 
     begin
	c_clk = 0;
	a_clk = 0;
	b_clk = 0;
	scan_in = 0;
	
     end
   
	
   always #100 c_clk = ~c_clk;
   
   initial
     begin

	reset = 1;

	
	
	#600 

	  reset = 0;
	req1_cmd = 0;
	req1_d1 = 0;
	req1_d2 = 0;
	req1_data = 0 ;
	req1_r1 = 0;
	req1_tag = 0;
	req2_cmd = 9;
	req2_d1 = 0;
	req2_d2 = 0;
	req2_data = 1 ;
	req2_r1 = 1;
	req2_tag = 0;
	req3_cmd = 0;
	req3_d1 = 0;
	req3_d2 = 0;
	req3_data = 0 ;
	req3_r1 = 0;
	req3_tag = 0;
	req4_cmd = 0;
	req4_d1 = 0;
	req4_d2 = 0;
	req4_data = 0 ;
	req4_r1 = 0;
	req4_tag = 0;
	
	#200 

	req2_cmd = 0;
	req2_d1 = 0;
	req2_d2 = 0;
	req2_data = 0 ;
	req2_r1 = 0;
	req2_tag = 0;
		
	#200

	req1_cmd = 9;
	req1_d1 = 0;
	req1_d2 = 0;
	req1_data = 2 ;
	req1_r1 = 2;
	req1_tag = 0;

	req2_cmd = 9;
	req2_d1 = 0;
	req2_d2 = 0;
	req2_data = 5 ;
	req2_r1 = 5;
	req2_tag = 1;

	#200
  
	req1_cmd = 0;
	req1_d1 = 0;
	req1_d2 = 0;
	req1_data = 0 ;
	req1_r1 = 0;
	req1_tag = 0;
	req2_cmd = 0;
	req2_d1 = 0;
	req2_d2 = 0;
	req2_data = 0 ;
	req2_r1 = 0;
	req2_tag = 0;

	#200

	  req1_cmd = 1;
	req1_d1 = 1;
	req1_d2 = 5;
	req1_data = 0 ;
	req1_r1 = 7;
	req1_tag = 2;

	#200
	
	req2_cmd = 0;
	req2_d1 = 0;
	req2_d2 = 0;
	req2_data = 0 ;
	req2_r1 = 0;
	req2_tag = 0;
		
	#200

	req2_cmd = 10;
	req2_d1 = 7;
	req2_d2 = 0;
	req2_data = 0 ;
	req2_r1 = 0;
	req2_tag = 3;

	#200

	req2_cmd = 0;
	req2_d1 = 0;
	req2_d2 = 0;
	req2_data = 0 ;
	req2_r1 = 0;
	req2_tag = 0;

	#2000 $stop;

     end // initial begin

   always
     @ (reset or c_clk or req1_cmd or req1_data or req2_cmd or req2_data or req3_cmd or req3_data or req4_cmd or req4_data) begin
	
	$display ("%t: r:%b \n 
1c:%d,1d1:%d,1d2:%d,1d:%d,1r:%d,1t:%d
2c:%d,2d1:%d,2d2:%d,2d:%d,2r:%d,2t:%d
3c:%d,3d1:%d,3d2:%d,3d:%d,3r:%d,3t:%d
4c:%d,4d1:%d,4d2:%d,4d:%d,4r:%d,4t:%d  
1r:%d,1d:%d,1t:%d  
2r:%d,2d:%d,2t:%d  
3r:%d,3d:%d,3t:%d  
4r:%d,4d:%d,4t:%d \n\n", 

$time, reset, 
req1_cmd, req1_d1, req1_d2, req1_data, req1_r1, req1_tag,
req2_cmd, req2_d1, req2_d2, req2_data, req2_r1, req2_tag,
req3_cmd, req3_d1, req3_d2, req3_data, req3_r1, req3_tag,
req4_cmd, req4_d1, req4_d2, req4_data, req4_r1, req4_tag,
out1_resp, out1_data, out1_tag,
out2_resp, out2_data, out2_tag,
out3_resp, out3_data, out3_tag,
out4_resp, out4_data, out4_tag);
	
     end


endmodule // test12


module atest13;

   wire [0:1] out1_resp, out1_tag, out2_resp, out2_tag, out3_resp, out3_tag, out4_resp, out4_tag;
   wire [0:31] out1_data, out2_data, out3_data, out4_data;
   wire 	 scan_out;
   
   reg [0:1] req1_tag, req2_tag, req3_tag, req4_tag;
   reg       a_clk, b_clk, c_clk, reset, scan_in;
   reg [0:3] req1_cmd, req1_d1, req1_d2, req1_r1, req2_cmd, req2_d1, req2_d2, req2_r1, req3_cmd, req3_d1, req3_d2, req3_r1, req4_cmd, req4_d1, req4_d2, req4_r1;
   reg [0:31] req1_data, req2_data, req3_data, req4_data;
      
   wire [0:15] add_shift_branch_data;
   wire [0:3]  adder_cmd, adder_out_cmd;
   wire [0:4]  adder_follow_branch;
   wire [0:31] adder_out_data1, adder_out_data2, adder_out_data3, adder_out_data4;
   wire [0:1]  adder_out_resp1, adder_out_resp2, adder_out_resp3, adder_out_resp4, adder_out_tag1, adder_out_tag2, adder_out_tag3, adder_out_tag4;
   wire        adder_overflow;
   wire [0:3]  adder_read_adr1, adder_read_adr2; 
   wire [0:63] adder_read_d1, adder_read_d2; 
   wire        adder_read_valid1, adder_read_valid2;
   wire [0:63] adder_result;
   wire [0:4]  adder_result_reg;
   wire [0:3]  adder_tag, adder_write_adr;
   wire [0:31] adder_write_data;
   wire        adder_write_valid;
   wire [0:3]  hold1_cmd, hold1_d1, hold1_d2;
   wire [0:31] hold1_data;
   wire [0:3]  hold1_r1;
   wire [0:1]  hold1_tag;
   wire [0:3]  hold2_cmd, hold2_d1, hold2_d2;
   wire [0:31] hold2_data;
   wire [0:3]  hold2_r1;
   wire [0:1]  hold2_tag;
   wire [0:3]  hold3_cmd, hold3_d1, hold3_d2;
   wire [0:31] hold3_data;
   wire [0:3]  hold3_r1;
   wire [0:1]  hold3_tag;
   wire [0:3]  hold4_cmd;
   wire [0:3]  hold4_d1,  hold4_d2;
   wire [0:31] hold4_data;
   wire [0:3]  hold4_r1;
   wire [0:1]  hold4_tag;
   wire port1_invalid_op;
   wire [0:1] port1_invalid_tag;
   wire       port2_invalid_op;
   wire [0:1] port2_invalid_tag;
   wire       port3_invalid_op;
   wire [0:1] port3_invalid_tag;
   wire       port4_invalid_op;
   wire [0:1] port4_invalid_tag;
   wire [0:3] prio_adder_cmd;
   wire [0:4] prio_adder_data1, prio_adder_data2;
   wire [0:4] prio_adder_follow_branch;
   wire       prio_adder_out_vld;
   wire [0:4] prio_adder_result;
   wire [0:3] prio_adder_tag,  prio_shift_cmd;
   wire [0:31] prio_shift_data;
   wire [0:4]  prio_shift_data1,  prio_shift_data2;
   wire [0:4]  prio_shift_follow_branch;
   wire        prio_shift_out_vld;
   wire [0:4]  prio_shift_result;
   wire [0:3]  prio_shift_tag;
   wire        scan_connect1, scan_connect10, scan_connect11, scan_connect12, scan_connect2, scan_connect3, scan_connect4, scan_connect5, scan_connect6, scan_connect7, scan_connect8, scan_connect9;
   wire [0:3]  shift_cmd;
   wire [0:4]  shift_follow_branch;
   wire [0:3]  shift_out_cmd;
   wire [0:31] shift_out_data1, shift_out_data2, shift_out_data3, shift_out_data4;
   wire [0:1]  shift_out_resp1;
   wire [0:1]  shift_out_resp2, shift_out_resp3, shift_out_resp4, shift_out_tag1, shift_out_tag2, shift_out_tag3, shift_out_tag4;
   wire [0:3]  shift_read_adr1,  shift_read_adr2;
   wire [0:63] shift_read_d1, shift_read_d2;
   wire        shift_read_valid1,  shift_read_valid2;
   wire [0:63] shift_result;
   wire [0:4]  shift_result_reg;
   wire [0:3]  shift_tag, shift_write_adr;
   wire [0:31] shift_write_data;
   wire        shift_write_valid, store_data_valid;
   wire [0:63] store_val;
   
   calc3_top ( out1_data, out1_resp, out1_tag, out2_data, out2_resp, out2_tag, out3_data, out3_resp, out3_tag, out4_data, out4_resp, out4_tag, scan_out, a_clk, b_clk, c_clk, req1_cmd, req1_d1, req1_d2, req1_data, req1_r1, req1_tag, req2_cmd, req2_d1, req2_d2, req2_data, req2_r1, req2_tag, req3_cmd, req3_d1, req3_d2, req3_data, req3_r1, req3_tag, req4_cmd, req4_d1, req4_d2, req4_data, req4_r1, req4_tag, reset, scan_in);
      
   initial 
     begin
	c_clk = 0;
	a_clk = 0;
	b_clk = 0;
	scan_in = 0;
	
     end
   
	
   always #100 c_clk = ~c_clk;
   
   initial
     begin

	reset = 1;

	
	
	#600 

	  reset = 0;
	req1_cmd = 12;
	req1_d1 = 15;
	req1_d2 = 0;
	req1_data = 0 ;
	req1_r1 = 0;
	req1_tag = 1;
	req2_cmd = 9;
	req2_d1 = 0;
	req2_d2 = 0;
	req2_data = 5 ;
	req2_r1 = 5;
	req2_tag = 3;
	req3_cmd = 9;
	req3_d1 = 0;
	req3_d2 = 0;
	req3_data = 1 ;
	req3_r1 = 1;
	req3_tag = 0;
	req4_cmd = 0;
	req4_d1 = 0;
	req4_d2 = 0;
	req4_data = 0 ;
	req4_r1 = 0;
	req4_tag = 0;
	
	#200 

	  req1_cmd = 0;
	req1_d1 = 0;
	req1_d2 = 0;
	req1_data = 0 ;
	req1_r1 = 0;
	req1_tag = 0;
	  
	req2_cmd = 0;
	req2_d1 = 0;
	req2_d2 = 0;
	req2_data = 0 ;
	req2_r1 = 0;
	req2_tag = 0;

	req3_cmd = 0;
	req3_d1 = 0;
	req3_d2 = 0;
	req3_data = 0 ;
	req3_r1 = 0;
	req3_tag = 0;

	req4_cmd = 0;
	req4_d1 = 0;
	req4_d2 = 0;
	req4_data = 0 ;
	req4_r1 = 0;
	req4_tag = 0;
	
	#200

	req1_cmd = 9;
	req1_d1 = 0;
	req1_d2 = 0;
	req1_data = 3 ;
	req1_r1 = 3;
	req1_tag = 3;

	req2_cmd = 9;
	req2_d1 = 0;
	req2_d2 = 0;
	req2_data = 2 ;
	req2_r1 = 2;
	req2_tag = 1;

	req3_cmd = 9;
	req3_d1 = 0;
	req3_d2 = 0;
	req3_data = 2;
	req3_r1 = 2;
	req3_tag = 1;

	req4_cmd = 0;
	req4_d1 = 0;
	req4_d2 = 0;
	req4_data = 0 ;
	req4_r1 = 0;
	req4_tag = 0;
		
	#200

	req1_cmd = 0;
	req1_d1 = 0;
	req1_d2 = 0;
	req1_data = 0 ;
	req1_r1 = 0;
	req1_tag = 0;
	req2_cmd = 0;
	req2_d1 = 0;
	req2_d2 = 0;
	req2_data = 0 ;
	req2_r1 = 0;
	req2_tag = 0;
	req3_cmd = 0;
	req3_d1 = 0;
	req3_d2 = 0;
	req3_data = 0 ;
	req3_r1 = 0;
	req3_tag = 0;
	req4_cmd = 0;
	req4_d1 = 0;
	req4_d2 = 0;
	req4_data = 0 ;
	req4_r1 = 0;
	req4_tag = 0;

	#200

	  req1_cmd = 9;
	req1_d1 = 0;
	req1_d2 = 0;
	req1_data = 15 ;
	req1_r1 = 15;
	req1_tag = 0;
	
	req2_cmd = 1;
	req2_d1 = 5;
	req2_d2 = 6;
	req2_data = 0 ;
	req2_r1 = 5;
	req2_tag = 2;
	
	req3_cmd = 1;
	req3_d1 = 1;
	req3_d2 = 2;
	req3_data = 0 ;
	req3_r1 = 1;
	req3_tag = 2;
	
	req4_cmd = 0;
	req4_d1 = 0;
	req4_d2 = 0;
	req4_data = 0 ;
	req4_r1 = 0;
	req4_tag = 0;
		
	#200

	  req1_cmd = 0;
	req1_d1 = 0;
	req1_d2 = 0;
	req1_data = 0 ;
	req1_r1 = 0;
	req1_tag = 0;
	req2_cmd = 0;
	req2_d1 = 0;
	req2_d2 = 0;
	req2_data = 0 ;
	req2_r1 = 0;
	req2_tag = 0;
	req3_cmd = 0;
	req3_d1 = 0;
	req3_d2 = 0;
	req3_data = 0 ;
	req3_r1 = 0;
	req3_tag = 0;
	req4_cmd = 0;
	req4_d1 = 0;
	req4_d2 = 0;
	req4_data = 0 ;
	req4_r1 = 0;
	req4_tag = 0; 

	#200

	  req1_cmd = 12;
	req1_d1 = 15;
	req1_d2 = 0;
	req1_data = 0 ;
	req1_r1 = 0;
	req1_tag = 1;
	req2_cmd = 10;
	req2_d1 = 5;
	req2_d2 = 0;
	req2_data = 0 ;
	req2_r1 = 0;
	req2_tag = 0;
	req3_cmd = 10;
	req3_d1 = 1;
	req3_d2 = 0;
	req3_data = 0 ;
	req3_r1 = 0;
	req3_tag = 3;
	req4_cmd = 0;
	req4_d1 = 0;
	req4_d2 = 0;
	req4_data = 0 ;
	req4_r1 = 0;
	req4_tag = 0;

	#200

	  req1_cmd = 0;
	req1_d1 = 0;
	req1_d2 = 0;
	req1_data = 0 ;
	req1_r1 = 0;
	req1_tag = 0;
	req2_cmd = 0;
	req2_d1 = 0;
	req2_d2 = 0;
	req2_data = 0 ;
	req2_r1 = 0;
	req2_tag = 0;
	req3_cmd = 0;
	req3_d1 = 0;
	req3_d2 = 0;
	req3_data = 0 ;
	req3_r1 = 0;
	req3_tag = 0;
	req4_cmd = 0;
	req4_d1 = 0;
	req4_d2 = 0;
	req4_data = 0 ;
	req4_r1 = 0;
	req4_tag = 0; 


	
	#2000 $stop;

     end // initial begin

   always
     @ (reset or c_clk or req1_cmd or req1_data or req2_cmd or req2_data or req3_cmd or req3_data or req4_cmd or req4_data) begin
	
	$display ("%t: r:%b \n 
1c:%d,1d1:%d,1d2:%d,1d:%d,1r:%d,1t:%d
2c:%d,2d1:%d,2d2:%d,2d:%d,2r:%d,2t:%d
3c:%d,3d1:%d,3d2:%d,3d:%d,3r:%d,3t:%d
4c:%d,4d1:%d,4d2:%d,4d:%d,4r:%d,4t:%d  
1r:%d,1d:%d,1t:%d  
2r:%d,2d:%d,2t:%d  
3r:%d,3d:%d,3t:%d  
4r:%d,4d:%d,4t:%d \n\n", 

$time, reset, 
req1_cmd, req1_d1, req1_d2, req1_data, req1_r1, req1_tag,
req2_cmd, req2_d1, req2_d2, req2_data, req2_r1, req2_tag,
req3_cmd, req3_d1, req3_d2, req3_data, req3_r1, req3_tag,
req4_cmd, req4_d1, req4_d2, req4_data, req4_r1, req4_tag,
out1_resp, out1_data, out1_tag,
out2_resp, out2_data, out2_tag,
out3_resp, out3_data, out3_tag,
out4_resp, out4_data, out4_tag);
	
     end


endmodule // test13
