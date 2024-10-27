// Library:  calc3
// Module:  Shifter Input Stage
// Author:  Naseer Siddique

module shift_input_stage( scan_out, shift_cmd, shift_follow_branch, shift_out_cmd, shift_read_adr1, shift_read_adr2, shift_read_valid1, shift_read_valid2, shift_result_reg, shift_tag, store_data_valid, store_val, a_clk, b_clk, c_clk, prio_shift_cmd, prio_shift_data1, prio_shift_data2, prio_shift_data, prio_shift_follow_branch, prio_shift_out_vld, prio_shift_result, prio_shift_tag, reset, scan_in);

   output [0:3] shift_cmd, shift_out_cmd, shift_read_adr1, shift_read_adr2, shift_tag;

   input [0:3] prio_shift_cmd, prio_shift_tag;

   input [0:31] prio_shift_data;

   output [0:4] shift_follow_branch, shift_result_reg;

   input [0:4] 	prio_shift_data1, prio_shift_data2, prio_shift_follow_branch, prio_shift_result;

   output 	shift_read_valid1, shift_read_valid2, scan_out, store_data_valid;

   output [0:63] store_val;
      
   input 	a_clk, b_clk, c_clk, prio_shift_out_vld, reset, scan_in;

   reg 		valid, data_valid;
   reg [0:3] 	cmd, out_cmd, tag;
   reg [0:4] 	d1, d2, follow_branch, result;
   reg [0:31] 	data;
   
   always
     @ (negedge c_clk) begin
	cmd <= (reset || ~prio_shift_out_vld) ? 4'b0 :
	       ((prio_shift_cmd[0:3] == 4'b0101) || (prio_shift_cmd[0:3] == 4'b1001)) ? 4'b0101 :
	       ((prio_shift_cmd[0:3] == 4'b0110)  || (prio_shift_cmd[0:3] == 4'b1010)) ? 4'b0110 :
	       4'b0;

	out_cmd <= (reset || ~prio_shift_out_vld) ? 4'b0 :
		   ((prio_shift_cmd[0:3] == 4'b0101) || (prio_shift_cmd == 4'b0110) || (prio_shift_cmd[0:3] == 4'b1001) || (prio_shift_cmd[0:3] == 4'b1010)) ? prio_shift_cmd[0:3] :
		   4'b0;
	

	d1 <= (reset) ? 5'b0 : prio_shift_data1;
	d2 <= (reset) ? 5'b0 : prio_shift_data2;
	follow_branch <= (reset) ? 5'b0 : prio_shift_follow_branch;
	valid <= (reset) ? 1'b0 : prio_shift_out_vld;
	result <= (reset) ? 5'b0 : prio_shift_result;
	tag <= (reset) ? 4'b0 : prio_shift_tag;
	data <= (reset) ? 32'b0 : prio_shift_data;
	data_valid <= (reset) ? 1'b0 : (prio_shift_cmd[0:3] == 4'b1001) ? 1'b1 : 1'b0;
		
     end // always @ (negedge c_clk)

   assign shift_follow_branch = follow_branch[0:4];
   assign shift_read_adr1 = d1[1:4];
   assign shift_read_adr2 = d2[1:4];
   assign shift_read_valid1 = d1[0];
   assign shift_read_valid2 = d2[0];
   assign shift_result_reg = result[0:4];
   assign shift_tag = tag[0:3];
   assign shift_cmd = cmd[0:3];
   assign shift_out_cmd = out_cmd[0:3];
   assign store_val = { 32'b0, data[0:31] };
   assign store_data_valid = data_valid;
   
   

endmodule // shift_input_stage
