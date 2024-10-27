// Library:  calc3
// Module:  Hold Register
// Author:  Naseer Siddique

module holdreg(hold_d1, hold_d2, hold_data, hold_prio_req, hold_prio_tag, hold_r1, scan_out, a_clk, b_clk, c_clk, req_cmd_in, req_d1, req_d2, req_data, req_r1, req_tag, reset, scan_in);
   
   output [0:3] hold_d1, hold_d2, hold_prio_req, hold_r1;
   output [0:31] hold_data;
   output [0:1]  hold_prio_tag;
   output 	 scan_out;

   input 	 a_clk, b_clk, c_clk, scan_in, reset;
   input [0:3] 	 req_cmd_in, req_d1, req_d2, req_r1;
   input [0:31]  req_data;
   input [0:1] 	req_tag;
   
   reg [0:3] 	hold_cmd_q, hold_data_reg1_q, hold_data_reg2_q, hold_result_reg_q;
   reg [0:1] 	hold_tag_q;
   reg [0:31] 	hold_data_q;
      
   always
     @ (negedge c_clk) begin
	hold_cmd_q[0:3] <= (reset) ? 4'b0 :
			   (req_cmd_in != 4'b0) ? req_cmd_in :
			   4'b0;

	hold_data_reg1_q <= (reset) ? 4'b0 :
			    (req_cmd_in != 4'b0) ? req_d1 :
			    4'b0;

	hold_data_reg2_q <= (reset) ? 4'b0 :
			    (req_cmd_in != 4'b0) ? req_d2 :
			    4'b0;

	hold_result_reg_q <= (reset) ? 4'b0 :
			     (req_cmd_in != 4'b0) ? req_r1 :
			     4'b0;

	hold_tag_q <= (reset) ? 2'b0 :
		      (req_cmd_in != 4'b0) ? req_tag :
		      2'b0;

	hold_data_q <= (reset) ? 32'b0 :
		       (req_cmd_in != 4'b0) ? req_data :
		       32'b0;

     end // always @ (negedge c_clk)

   assign hold_prio_req = hold_cmd_q;
   assign hold_d1 = hold_data_reg1_q;
   assign hold_d2 = hold_data_reg2_q;
   assign hold_r1 = hold_result_reg_q;
   assign hold_prio_tag = hold_tag_q;
   assign hold_data = hold_data_q;
         
endmodule // holdreg