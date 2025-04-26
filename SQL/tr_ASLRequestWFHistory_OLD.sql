IF  EXISTS (SELECT * FROM sys.triggers WHERE object_id = OBJECT_ID(N'[dbo].[tr_ASLRequestWFHistory]'))
DROP TRIGGER [dbo].[tr_ASLRequestWFHistory]
GO
SET ANSI_NULLS ON
GO

SET QUOTED_IDENTIFIER ON
GO


CREATE TRIGGER [dbo].[tr_ASLRequestWFHistory] ON [dbo].[ASLRequestWFHistory] 
FOR UPDATE,DELETE
AS
/*********************************************************************/
/*                                                                   */
/*********************************************************************/
SET NOCOUNT ON

/*********************************************************************/
/*                                                                   */
/*********************************************************************/
Declare		@Info	Varchar(25)

/*********************************************************************/
/*                                                                   */
/*********************************************************************/
If	(Select	Count(AU.EmpID) 
	From ASLRequestWFHistory AU INNER JOIN deleted DEL 
	ON AU.ID = DEL.ID 
	AND AU.EmpId = DEL.EmpID) > 0

	Begin
		Set	@Info =  'Update'	
	End
 Else
	Begin
		Set	@Info =  'Delete'
	End


/*********************************************************************/
/*                                                                   */
/*********************************************************************/
Insert	AuditASLRequestWFHistory
SELECT	ID,
		EmpId,
		MgrId,
		MgrApprovalDate,
		WorkFlowStatusID,
		MgrComments,
		LastUser,
		LastUpdate,
		ArchiveDate,
		@Info,
		GETDATE(),
		USER_NAME()
From	Deleted

/*********************************************************************/
/*                                                                   */
/*********************************************************************/
SET NOCOUNT OFF
GO


