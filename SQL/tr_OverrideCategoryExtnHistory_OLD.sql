IF  EXISTS (SELECT * FROM sys.triggers WHERE object_id = OBJECT_ID(N'[dbo].[tr_OverrideCategoryExtnHistory]'))
DROP TRIGGER [dbo].[tr_OverrideCategoryExtnHistory]
GO
SET ANSI_NULLS ON
GO

SET QUOTED_IDENTIFIER ON
GO


CREATE TRIGGER [dbo].[tr_OverrideCategoryExtnHistory] ON [dbo].[OverrideCategoryExtnHistory] 
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
If	(Select	Count(OC.EmpId) 
	From OverrideCategoryExtnHistory OC INNER JOIN deleted DEL 
	ON OC.EmpId = DEL.EmpId) > 0

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
Insert	AuditOverrideCategoryExtnHistory
SELECT	EmpId,
		Comment,
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