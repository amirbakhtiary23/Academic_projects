// Library:  calc3
// Module:  Shifter Output Stage
// Author:  Naseer Siddique

module shift_output_stage(scan_out, shift_out_data1, shift_out_data2, shift_out_data3, shift_out_data4, shift_out_resp1, shift_out_resp2, shift_out_resp3, shift_out_resp4, shift_out_tag1,  shift_out_tag2,  shift_out_tag3,  shift_out_tag4, shift_write_adr, shift_write_data, shift_write_valid, a_clk, add_shift_branch_data, b_clk, c_clk, reset, scan_in, shift_follow_branch, shift_out_cmd, shift_result, shift_result_reg, shift_tag);

   output scan_out, shift_write_valid;
   output [0:31] shift_out_data1, shift_out_data2, shift_out_data3, shift_out_data4, shift_write_data;
   output [0:1] 	 shift_out_resp1, shift_out_resp2, shift_out_resp3, shift_out_resp4, shift_out_tag1,  shift_out_tag2,  shift_out_tag3,  shift_out_tag4;
   output [0:3] 	 shift_write_adr;

   input 		 a_clk, b_clk, c_clk, reset, scan_in;
   input [0:15] 	 add_shift_branch_data;
   input [0:4] 		 shift_follow_branch, shift_result_reg;
   input [0:3] 		 shift_out_cmd, shift_tag;
   input [0:63] 	 shift_result;

   wire			 valid_cmd, skip_cmd;
   reg [0:1] 		 hold_resp;
   reg [0:3] 		 hold_tag;
   reg [0:4] 		 hold_result_reg;
   reg [0:31] 		 hold_reg_data, hold_out_data;

   assign 		 valid_cmd = ((shift_out_cmd == 4'b0101) || (shift_out_cmd == 4'b0110) || (shift_out_cmd == 4'b1001) || (shift_out_cmd == 4'b1010)) ? 1'b1 : 1'b0;

   assign 		 skip_cmd = ( shift_follow_branch[0] && (add_shift_branch_data[shift_follow_branch[1:4]]) ) ? 1'b1 : 1'b0;

   always
     @ (negedge c_clk) begin
	hold_result_reg[0:4] <= ( reset || ~valid_cmd || skip_cmd ) ? 5'b0 : shift_result_reg;

	hold_tag[0:3] <= (reset || ~valid_cmd) ? 4'b0 : shift_tag[0:3];

	hold_resp[0:1] <= ( reset || ~valid_cmd) ? 2'b0 : (skip_cmd) ? 2'b11 : 2'b01;

	hold_reg_data[0:31] <= (reset || ~valid_cmd || skip_cmd) ? 32'b0 : shift_result[32:63];

	hold_out_data[0:31] <= (reset || ~valid_cmd || skip_cmd || (shift_out_cmd != 4'b1010)) ? 32'b0 : shift_result[32:63];
     end // always @ (negedge c_clk)
				
   assign shift_write_adr[0:3] = hold_result_reg[1:4];
   assign shift_write_data[0:31] = hold_reg_data[0:31];
   assign shift_write_valid = hold_result_reg[0];

   assign shift_out_resp1 = (hold_tag[0:1] == 'b00) ? hold_resp : 'b00;
   assign shift_out_resp2 = (hold_tag[0:1] == 'b01) ? hold_resp : 'b00;
   assign shift_out_resp3 = (hold_tag[0:1] == 'b10) ? hold_resp : 'b00;
   assign shift_out_resp4 = (hold_tag[0:1] == 'b11) ? hold_resp : 'b00;

   assign shift_out_data1 = (hold_tag[0:1] == 'b00) ? hold_out_data : 32'b0;
   assign shift_out_data2 = (hold_tag[0:1] == 'b01) ? hold_out_data : 32'b0;
   assign shift_out_data3 = (hold_tag[0:1] == 'b10) ? hold_out_data : 32'b0;
   assign shift_out_data4 = (hold_tag[0:1] == 'b11) ? hold_out_data : 32'b0;
   
   assign shift_out_tag1 = (hold_tag[0:1] == 'b00) ? hold_tag[2:3] : 'b00;
   assign shift_out_tag2 = (hold_tag[0:1] == 'b01) ? hold_tag[2:3] : 'b00;
   assign shift_out_tag3 = (hold_tag[0:1] == 'b10) ? hold_tag[2:3] : 'b00;
   assign shift_out_tag4 = (hold_tag[0:1] == 'b11) ? hold_tag[2:3] : 'b00;
   
endmodule // shift_output_stage
